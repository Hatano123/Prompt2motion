#!/usr/bin/env python3
"""Build a cross-demo Phase B table from completed held-out tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def rollout_file(directory):
    matches = list(directory.glob("rollouts_*.jsonl"))
    return matches[0] if len(matches) == 1 else None


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def discover():
    records = []
    roots = [("phase_b", ROOT / "results/act/phase_b"),
             ("phase_b_rebuild", ROOT / "results/act/phase_b_rebuild")]
    for namespace, base in roots:
        if not base.exists(): continue
        for test_dir in base.glob("*demos/*/train_seed*/eval_test"):
            if not (test_dir / ".complete").exists(): continue
            path = rollout_file(test_dir)
            if path is None: continue
            run = test_dir.parent
            demos = int(run.parents[1].name.removesuffix("demos"))
            model = run.parent.name
            seed = int(run.name.removeprefix("train_seed"))
            rows = load_rows(path)
            state_ids = [int(row.get("rollout_id", row.get("rollout"))) for row in rows]
            if state_ids != list(range(100)):
                raise ValueError(f"non-canonical rollout IDs: {path}")
            records.append({"namespace": namespace, "demo_count": demos, "model": model,
                            "training_seed": seed, "rollouts": len(rows),
                            "successes": sum(bool(row["success"]) for row in rows),
                            "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
                            "average_return": statistics.mean(float(row["episode_return"]) for row in rows),
                            "rollout_file": str(path.relative_to(ROOT))})
    return sorted(records, key=lambda row: (row["demo_count"], row["model"], row["training_seed"]))


def aggregate(records):
    groups = {}
    for row in records: groups.setdefault((row["demo_count"], row["model"]), []).append(row)
    result = []
    for (demos, model), rows in sorted(groups.items()):
        rates = [row["success_rate"] for row in rows]
        result.append({"demo_count": demos, "model": model, "completed_seeds": len(rows),
                       "seed_mean_success_rate": statistics.mean(rates),
                       "seed_sd_success_rate": statistics.stdev(rates) if len(rates) > 1 else None,
                       "mean_return": statistics.mean(row["average_return"] for row in rows)})
    return result


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists(): raise SystemExit(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    records = discover(); groups = aggregate(records)
    write_csv(output / "runs.csv", records); write_csv(output / "summary.csv", groups)
    (output / "summary.json").write_text(json.dumps({"runs": records, "groups": groups}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "completed_runs": len(records), "groups": len(groups)}))


if __name__ == "__main__": main()
