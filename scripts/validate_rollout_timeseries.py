#!/usr/bin/env python3
"""Validate detailed ACT rollout summaries and compressed timestep logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "repos" / "act"))

from rollout_logging import validate_episode_logs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path,
                        help="A rollout summary JSON or a directory containing *_summary.json files")
    parser.add_argument("--manifest", type=Path,
                        help="Optional evaluation manifest used to verify initial states")
    args = parser.parse_args()

    if args.path.is_dir():
        paths = sorted(args.path.glob("rollout_*_summary.json"))
    else:
        paths = [args.path]
    if not paths:
        raise SystemExit(f"No rollout summaries found under {args.path}")

    manifest_states = None
    if args.manifest:
        manifest_states = np.asarray(json.loads(args.manifest.read_text())["initial_states"])

    total_bytes = 0
    for path in paths:
        expected = None
        if manifest_states is not None:
            rollout_id = int(path.name.split("_")[1])
            expected = manifest_states[rollout_id]
        result = validate_episode_logs(path, expected_initial_state=expected)
        total_bytes += result["compressed_bytes"]
        print(json.dumps(result, sort_keys=True))
    print(json.dumps({
        "valid_rollouts": len(paths),
        "compressed_bytes": total_bytes,
        "projected_100_rollouts_bytes": round(total_bytes / len(paths) * 100),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
