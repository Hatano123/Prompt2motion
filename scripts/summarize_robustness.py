#!/usr/bin/env python3
"""Summarize clean/corrupted paired rollouts without ACT dependencies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metrics(rows):
    successes = [bool(row["success"]) for row in rows]
    returns = [float(row["episode_return"]) for row in rows]
    return {"rollouts": len(rows), "successes": sum(successes),
            "success_rate": sum(successes) / len(rows), "average_return": statistics.mean(returns)}


def exact_mcnemar(discordant_a, discordant_b):
    n = discordant_a + discordant_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(discordant_a, discordant_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_stats(clean, corrupt, bootstrap_samples=10000):
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    clean_ids = [int(row.get("rollout_id", row.get("rollout"))) for row in clean]
    corrupt_ids = [int(row.get("rollout_id", row.get("rollout"))) for row in corrupt]
    if len(set(clean_ids)) != len(clean_ids) or len(set(corrupt_ids)) != len(corrupt_ids):
        raise ValueError("paired rollout IDs must be unique")
    clean_by_id = dict(zip(clean_ids, clean))
    corrupt_by_id = dict(zip(corrupt_ids, corrupt))
    ids = sorted(clean_by_id)
    if not ids or ids != list(range(len(ids))) or set(clean_by_id) != set(corrupt_by_id):
        raise ValueError("paired rollout IDs must be contiguous and identical")
    differences = [int(bool(corrupt_by_id[i]["success"])) - int(bool(clean_by_id[i]["success"])) for i in ids]
    clean_only = sum(bool(clean_by_id[i]["success"]) and not bool(corrupt_by_id[i]["success"]) for i in ids)
    corrupt_only = sum(not bool(clean_by_id[i]["success"]) and bool(corrupt_by_id[i]["success"]) for i in ids)
    rng = random.Random(2401)
    boot = []
    for _ in range(bootstrap_samples):
        boot.append(sum(differences[rng.randrange(len(ids))] for _ in ids) / len(ids))
    boot.sort()
    return {"paired_rollouts": len(ids), "success_rate_delta": sum(differences) / len(ids),
            "delta_ci95_low": boot[int(0.025 * bootstrap_samples)],
            "delta_ci95_high": boot[int(0.975 * bootstrap_samples) - 1],
            "clean_success_corrupt_failure": clean_only,
            "clean_failure_corrupt_success": corrupt_only,
            "mcnemar_exact_p": exact_mcnemar(clean_only, corrupt_only)}


def find_rollouts(directory):
    matches = sorted(directory.glob("rollouts_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one rollout JSONL in {directory}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="A train_seed directory containing condition subdirectories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    condition_dirs = sorted(path for path in args.root.iterdir() if path.is_dir() and (path / ".complete").exists())
    clean_dir = next((path for path in condition_dirs if path.name == "clean"), None)
    clean_rows = read_jsonl(find_rollouts(clean_dir)) if clean_dir else None
    output_rows = []
    for directory in condition_dirs:
        rows = read_jsonl(find_rollouts(directory))
        row = {"condition": directory.name, **metrics(rows)}
        if clean_rows is not None and directory != clean_dir:
            row.update(paired_stats(clean_rows, rows))
        output_rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in output_rows for key in row})
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(output_rows)
    (args.output / "summary.json").write_text(json.dumps(output_rows, indent=2, sort_keys=True) + "\n")
    lines = ["# Robustness summary", "", "| condition | success | delta vs clean | 95% CI | McNemar p |",
             "|---|---:|---:|---:|---:|"]
    for row in output_rows:
        delta = row.get("success_rate_delta")
        ci = "—" if delta is None else f"[{row['delta_ci95_low']:.3f}, {row['delta_ci95_high']:.3f}]"
        delta_text = "—" if delta is None else f"{delta:+.3f}"
        p_text = "—" if delta is None else f"{row['mcnemar_exact_p']:.4g}"
        lines.append(f"| {row['condition']} | {row['successes']}/{row['rollouts']} ({row['success_rate']:.3f}) | "
                     f"{delta_text} | {ci} | {p_text} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
