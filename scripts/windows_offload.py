#!/usr/bin/env python3
"""Prepare and commit a verified Windows pull of one completed Phase B run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import sequential_experiment_runner as runner

MANIFEST = "windows_offload_manifest.json"
RECEIPT = "offload_windows_receipt.json"


def resolve_run(name: str) -> Path:
    if not runner.RUN_RE.fullmatch(name):
        raise ValueError(f"unsupported run: {name}")
    return runner.run_dir(name)


def prepare(name: str) -> dict:
    if not runner.no_active_jobs():
        raise ValueError("PBS jobs are active; wait for a stable run")
    run = resolve_run(name)
    for stage in ("train", "select", "test"):
        if not runner.verify_stage(stage, run):
            raise ValueError(f"run has not passed {stage}: {name}")
    path = run / MANIFEST
    files = [row for row in runner.inventory(run) if row["path"] not in (MANIFEST, RECEIPT)]
    if not files:
        raise ValueError("run has no files to transfer")
    if path.exists():
        result = json.loads(runner.read_text(path))
        if result.get("run") != name or result.get("files") != files:
            raise ValueError("run changed after Windows offload preparation")
    else:
        result = {"schema_version": 1, "run": name, "files": files,
                  "total_bytes": sum(row["bytes"] for row in files)}
        runner.atomic_json(path, result)
    return {**result, "manifest_sha256": runner.digest(path)}


def commit(name: str, receipt: dict, prune: bool = True) -> dict:
    if not runner.no_active_jobs():
        raise ValueError("PBS jobs are active; refusing offload commit")
    run = resolve_run(name)
    manifest_path = run / MANIFEST
    manifest = json.loads(runner.read_text(manifest_path))
    if manifest.get("schema_version") != 1 or manifest.get("run") != name:
        raise ValueError("invalid Windows offload manifest")
    expected = {
        "schema_version": 1, "run": name, "verified": True,
        "manifest_sha256": runner.digest(manifest_path),
        "file_count": len(manifest["files"]), "total_bytes": manifest["total_bytes"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Windows receipt does not match the prepared manifest")
    destination = receipt.get("destination")
    if not isinstance(destination, str) or not destination:
        raise ValueError("Windows destination missing")
    selection = json.loads(runner.read_text(run / "checkpoint_selection_deterministic/validation_summary.json"))
    selected = selection["selected_checkpoint"]
    candidates = json.loads(runner.read_text(run / "checkpoint_manifest.json"))["checkpoints"]
    archived = [row for row in candidates if row["name"] != selected]
    listed = {row["path"]: row for row in manifest["files"]}
    for row in archived:
        saved = listed.get(row["name"])
        if saved is None or saved["bytes"] != row["bytes"] or saved["sha256"] != row["sha256"]:
            raise ValueError(f"candidate absent from Windows manifest: {row['name']}")
    receipt = {**expected, "destination": destination, "archived_checkpoints": archived}
    receipt_path = run / RECEIPT
    if receipt_path.exists():
        if json.loads(runner.read_text(receipt_path)) != receipt:
            raise ValueError("existing Windows receipt differs")
    else:
        # The client verified all files before sending this receipt. Check that
        # the source still matches the snapshot before deleting any checkpoint.
        runner.verify_copy(run, manifest["files"])
        runner.atomic_json(receipt_path, receipt)
    if prune:
        for row in archived:
            source = run / row["name"]
            if source.exists():
                if source.stat().st_size != row["bytes"] or runner.digest(source) != row["sha256"]:
                    raise ValueError(f"checkpoint changed before prune: {source}")
                source.unlink()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "commit"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args()
    result = prepare(args.run) if args.action == "prepare" else commit(
        args.run, json.load(sys.stdin), prune=not args.keep_checkpoints
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except runner.InfrastructureUnavailable as exc:
        print(f"[WAIT_INFRA] {exc}", file=sys.stderr)
        sys.exit(75)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
