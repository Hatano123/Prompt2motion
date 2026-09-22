#!/usr/bin/env python3
"""Compare lightweight ACT behavior traces from logging-OFF and logging-ON runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("off_dir", type=Path)
    parser.add_argument("on_dir", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    off_paths = sorted(args.off_dir.glob("rollout_*_behavior.npz"))
    on_paths = sorted(args.on_dir.glob("rollout_*_behavior.npz"))
    if not off_paths or [path.name for path in off_paths] != [path.name for path in on_paths]:
        raise SystemExit("OFF/ON behavior trace sets are empty or do not match")

    fields = ("timestep", "initial_state", "observed_qpos", "action", "reward", "success")
    compared = []
    for off_path, on_path in zip(off_paths, on_paths):
        with np.load(off_path, allow_pickle=False) as off, np.load(on_path, allow_pickle=False) as on:
            for field in fields:
                if field not in off.files or field not in on.files:
                    raise SystemExit(f"Missing {field} in {off_path.name}")
                if off[field].shape != on[field].shape:
                    raise SystemExit(
                        f"Behavior shape mismatch: {off_path.name} field={field} "
                        f"{off[field].shape} != {on[field].shape}")
                if not np.allclose(off[field], on[field], atol=args.atol, rtol=args.rtol,
                                   equal_nan=False):
                    maximum = float(np.max(np.abs(
                        off[field].astype(np.float64) - on[field].astype(np.float64))))
                    raise SystemExit(
                        f"Behavior mismatch: {off_path.name} field={field} max_abs={maximum}")
        compared.append(off_path.name)
    print(json.dumps({
        "behavior_identical": True,
        "rollouts_compared": len(compared),
        "fields": list(fields),
        "atol": args.atol,
        "rtol": args.rtol,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
