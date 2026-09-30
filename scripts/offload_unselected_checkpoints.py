#!/usr/bin/env python3
"""Move unselected checkpoints out of home after verified Drive archival.

The selected checkpoint is always retained. Existing files are never
overwritten, and every candidate is checked against checkpoint_manifest.json
before a receipt is created or a file is moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="repository-relative completed training run")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--staging-root", type=Path, default=Path("/tmp/Prompt2Motion_offloaded_candidates"))
    parser.add_argument("--drive-folder-id", required=True)
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--archive-parts-checksum-file-id", required=True)
    parser.add_argument("--drive-readback-verified", action="store_true", required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    run = (root / args.run_dir).resolve()
    try:
        relative_run = run.relative_to(root)
    except ValueError as exc:
        raise SystemExit("run directory must be inside repository root") from exc
    if not run.is_dir():
        raise SystemExit(f"run directory is missing: {run}")

    selection_dir = run / "checkpoint_selection_deterministic"
    if not (selection_dir / ".complete").exists():
        raise SystemExit("deterministic checkpoint selection is incomplete")
    selection = json.loads((selection_dir / "validation_summary.json").read_text())
    if selection.get("selection_dataset") != "fixed_validation_manifest":
        raise SystemExit("selection did not use the fixed validation manifest")
    if selection.get("heldout_test_used_for_selection") is not False:
        raise SystemExit("held-out test was used for selection")
    selected = selection["selected_checkpoint"]

    manifest = json.loads((run / "checkpoint_manifest.json").read_text())
    candidates = manifest.get("checkpoints", [])
    if len(candidates) != 4:
        raise SystemExit("checkpoint manifest must contain exactly four candidates")
    names = {item["name"] for item in candidates}
    if selected not in names:
        raise SystemExit("selected checkpoint is absent from checkpoint manifest")

    staging_run = args.staging_root.resolve() / relative_run
    archived = []
    for item in candidates:
        path = run / item["name"]
        staged = staging_run / item["name"]
        source = path if path.is_file() else staged if staged.is_file() else None
        if source is None:
            raise SystemExit(f"candidate is unavailable locally: {item['name']}")
        if source.stat().st_size != item["bytes"] or sha256(source) != item["sha256"]:
            raise SystemExit(f"candidate failed manifest verification: {source}")
        if item["name"] != selected:
            archived.append(item)

    selected_path = run / selected
    if not selected_path.is_file():
        raise SystemExit("selected checkpoint must remain in the run directory")

    receipt = {
        "schema_version": 1,
        "archive_name": args.archive_name,
        "archive_parts_checksum_file_id": args.archive_parts_checksum_file_id,
        "drive_folder_id": args.drive_folder_id,
        "drive_readback_verified": True,
        "retained_selected_checkpoint": selected,
        "archived_checkpoints": archived,
        "offloaded_unix_seconds": time.time(),
    }
    receipt_path = run / "checkpoint_archive_receipt.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        for key in ("schema_version", "archive_name", "archive_parts_checksum_file_id",
                    "drive_folder_id", "drive_readback_verified",
                    "retained_selected_checkpoint", "archived_checkpoints"):
            if existing.get(key) != receipt.get(key):
                raise SystemExit(f"existing receipt differs at {key}; refusing to overwrite")
    else:
        temporary = receipt_path.with_name(f"{receipt_path.name}.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, receipt_path)

    staging_run.mkdir(parents=True, exist_ok=True)
    for item in archived:
        source = run / item["name"]
        destination = staging_run / item["name"]
        if destination.exists():
            if destination.stat().st_size != item["bytes"] or sha256(destination) != item["sha256"]:
                raise SystemExit(f"staged checkpoint differs: {destination}")
            if source.exists():
                raise SystemExit(f"both source and staging destination exist: {item['name']}")
            print(f"[SKIP] already offloaded: {item['name']}")
            continue
        if not source.is_file():
            raise SystemExit(f"source disappeared before move: {source}")
        shutil.move(str(source), str(destination))
        print(f"[OFFLOADED] {relative_run / item['name']}")

    print(f"[OK] retained selected checkpoint: {relative_run / selected}")


if __name__ == "__main__":
    main()
