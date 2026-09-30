#!/usr/bin/env python3
"""Create an immutable phase gate only after every required artifact passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiment_pipeline.json"


def source(stage, method, demos, model, seed):
    if stage == "phase_b":
        rebuild = ROOT / f"results/act/phase_b_rebuild/{demos}demos/{model}/train_seed{seed}"
        if (rebuild / ".training_complete").exists(): return rebuild
        return ROOT / f"results/act/phase_b/{demos}demos/{model}/train_seed{seed}"
    return ROOT / f"results/act/phase_d/{method}/{demos}demos/{model}/train_seed{seed}"


def require(report, path, description):
    if not path.exists(): report["errors"].append(f"missing {description}: {path.relative_to(ROOT)}")
    else: report["artifacts"].append(str(path.relative_to(ROOT)))


def archived_checkpoint_matches(run, item):
    receipt_path = run / "checkpoint_archive_receipt.json"
    if not receipt_path.is_file(): return False
    try:
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema_version") != 1 or receipt.get("drive_readback_verified") is not True:
            return False
        matches = [entry for entry in receipt.get("archived_checkpoints", [])
                   if entry.get("name") == item.get("name")]
        return len(matches) == 1 and all(matches[0].get(key) == item.get(key)
                                         for key in ("bytes", "sha256"))
    except (OSError, TypeError, json.JSONDecodeError):
        return False


def audit_training(report, run, demos):
    require(report, run / ".training_complete", "training marker")
    summary_path = run / "training_summary.json"
    require(report, summary_path, "training summary")
    if summary_path.is_file():
        data = json.loads(summary_path.read_text())
        if data.get("optimizer_steps") != 10000: report["errors"].append(f"wrong optimizer budget: {run.relative_to(ROOT)}")
        if data.get("sampled_episode_starts") != 80000: report["errors"].append(f"wrong sample budget: {run.relative_to(ROOT)}")
    split_path = run / "split_manifest.json"
    require(report, split_path, "split manifest")
    if split_path.is_file():
        split = json.loads(split_path.read_text())
        if split.get("train_episode_ids") != list(range(demos)): report["errors"].append(f"wrong nested train split: {run.relative_to(ROOT)}")
        if split.get("val_episode_ids") != list(range(40, 50)): report["errors"].append(f"wrong validation split: {run.relative_to(ROOT)}")
    checkpoint_manifest = run / "checkpoint_manifest.json"
    require(report, checkpoint_manifest, "checkpoint manifest")
    if checkpoint_manifest.is_file():
        for item in json.loads(checkpoint_manifest.read_text()).get("checkpoints", []):
            checkpoint = run / item["name"]
            if checkpoint.is_file():
                if checkpoint.stat().st_size != item.get("bytes"):
                    report["errors"].append(f"candidate checkpoint size mismatch: {checkpoint.relative_to(ROOT)}")
                elif hashlib.sha256(checkpoint.read_bytes()).hexdigest() != item.get("sha256"):
                    report["errors"].append(f"candidate checkpoint hash mismatch: {checkpoint.relative_to(ROOT)}")
                else:
                    report["artifacts"].append(str(checkpoint.relative_to(ROOT)))
            elif archived_checkpoint_matches(run, item):
                report["artifacts"].append(f"drive-archived:{(run / item['name']).relative_to(ROOT)}")
            else:
                report["errors"].append(f"missing candidate checkpoint: {checkpoint.relative_to(ROOT)}")


def condition_labels(model, conditions):
    model_cameras = {"single_top": ["top"], "single_side": ["side"], "multi": ["top", "side"]}[model]
    for condition in conditions:
        if condition == "clean": yield "clean"
        else:
            for camera in model_cameras: yield f"{condition}_{camera}"


def audit(phase, config):
    report = {"schema_version": 1, "phase": phase, "valid": False, "errors": [], "artifacts": [],
              "audited_unix_seconds": time.time()}
    seeds = config["training"]["seeds"]
    views = config["training"]["views"]
    if phase in ("b0", "b"):
        selected_seeds = [0] if phase == "b0" else seeds
        demo_counts = (config["phase_b"]["b0_demo_counts"] if phase == "b0"
                       else config["phase_b"]["new_training_demo_counts"])
        for demos in demo_counts:
            for model in views:
                for seed in selected_seeds:
                    run = source("phase_b", "naive", demos, model, seed)
                    audit_training(report, run, demos)
                    require(report, run / "checkpoint_selection_deterministic/.complete", "deterministic selection")
                    require(report, run / "eval_test/.complete", "held-out test")
                    require(report, run / "eval_test/timeseries/index.jsonl", "timeseries index")
    elif phase == "c":
        for demos in config["phase_c"]["demo_counts"]:
            for model in views:
                for seed in seeds:
                    base = ROOT / f"results/act/phase_c/{demos}demos/{model}/train_seed{seed}"
                    for label in condition_labels(model, config["phase_c"]["conditions"]):
                        require(report, base / label / ".complete", "robustness result")
                        require(report, base / label / "timeseries/index.jsonl", "timeseries index")
    elif phase == "d":
        for method in config["phase_d"]["methods"]:
            for demos in config["phase_d"]["demo_counts"]:
                for seed in seeds:
                    run = source("phase_d", method, demos, "multi", seed)
                    audit_training(report, run, demos)
                    require(report, run / "checkpoint_selection_deterministic/.complete", "deterministic selection")
                    require(report, run / "eval_test/.complete", "held-out clean test")
                    for label in condition_labels("multi", config["phase_c"]["conditions"]):
                        require(report, run / "eval_robustness" / label / ".complete", "method robustness result")
    else: raise ValueError(phase)
    report["valid"] = not report["errors"]
    report["config_sha256"] = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["b0", "b", "c", "d"], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    report = audit(args.phase, config)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered)
    if args.output:
        target = args.output if args.output.is_absolute() else ROOT / args.output
        if target.exists(): raise SystemExit(f"refusing to overwrite immutable gate: {target}")
        if not report["valid"]: raise SystemExit("phase is incomplete; gate was not written")
        target.parent.mkdir(parents=True, exist_ok=True); target.write_text(rendered)
    raise SystemExit(0 if report["valid"] else 2)


if __name__ == "__main__": main()
