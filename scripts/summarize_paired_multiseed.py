#!/usr/bin/env python3
"""Summarize completed clean/robustness rollouts without requiring ACT dependencies."""

import argparse
import csv
import json
import math
import os


def read_rollouts(path):
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def wilson(successes, count, z=1.959963984540054):
    if not count:
        return (float("nan"), float("nan"))
    rate = successes / count
    scale = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / scale
    return center - radius, center + radius


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/act/paired_pilot")
    parser.add_argument("--output", default="results/act/paired_pilot/multiseed_summary.csv")
    args = parser.parse_args()
    rows = []
    for model in ("single_top", "single_side", "multi"):
        for seed in (0, 1, 2):
            run_dir = os.path.join(args.root, model, f"train_seed{seed}")
            summary_path = os.path.join(run_dir, "pilot_summary.json")
            if not os.path.exists(summary_path):
                continue
            with open(summary_path, encoding="utf-8") as source:
                summary = json.load(source)
            selected_epoch = summary["selected_epoch"]
            candidates = [
                os.path.join(run_dir, "eval_clean", f"epoch_{selected_epoch}",
                             f"rollouts_policy_epoch_{selected_epoch}_seed_{seed}.jsonl"),
                os.path.join(run_dir, "artifacts", "eval_clean", f"epoch_{selected_epoch}",
                             f"rollouts_policy_epoch_{selected_epoch}_seed_{seed}.jsonl"),
            ]
            clean_path = next((path for path in candidates if os.path.exists(path)), None)
            paths = [("clean", clean_path)]
            robust_dir = os.path.join(run_dir, "eval_robustness")
            if os.path.isdir(robust_dir):
                for condition in sorted(os.listdir(robust_dir)):
                    path = os.path.join(robust_dir, condition, "rollouts_selected_policy.jsonl")
                    if os.path.exists(path):
                        paths.append((condition, path))
            clean_rate = None
            condition_data = []
            for condition, path in paths:
                if not path:
                    continue
                rollouts = read_rollouts(path)
                successes = sum(bool(item["success"]) for item in rollouts)
                rate = successes / len(rollouts)
                if condition == "clean":
                    clean_rate = rate
                condition_data.append((condition, successes, len(rollouts), rate))
            for condition, successes, count, rate in condition_data:
                low, high = wilson(successes, count)
                rows.append({"model": model, "training_seed": seed,
                             "selected_epoch": selected_epoch, "condition": condition,
                             "successes": successes, "rollouts": count,
                             "success_rate": rate, "ci95_low": low, "ci95_high": high,
                             "drop_from_clean": "" if clean_rate is None else clean_rate - rate})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fields = ["model", "training_seed", "selected_epoch", "condition", "successes",
              "rollouts", "success_rate", "ci95_low", "ci95_high", "drop_from_clean"]
    with open(args.output, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
