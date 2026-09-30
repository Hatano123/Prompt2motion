#!/usr/bin/env python3
"""Inventory and optionally submit only currently eligible Phase B--D jobs.

Dry-run is the default.  The planner never removes, replaces, or truncates an
experiment artifact.  Submissions are appended to a registry and are blocked
when the configured filesystem reserve would be crossed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment_pipeline.json"


def cameras(model: str) -> list[str]:
    return {"single_top": ["top"], "single_side": ["side"], "multi": ["top", "side"]}[model]


def source_dir(stage: str, method: str, demos: int, model: str, seed: int) -> Path:
    if stage == "phase_b":
        canonical = ROOT / f"results/act/phase_b/{demos}demos/{model}/train_seed{seed}"
        rebuild = ROOT / f"results/act/phase_b_rebuild/{demos}demos/{model}/train_seed{seed}"
        if (canonical / "checkpoint_selection_deterministic/.complete").exists() and selected_checkpoint(canonical):
            return canonical
        if (canonical / ".training_complete").exists() and not candidate_checkpoints_available(canonical):
            return rebuild
        return canonical
    return ROOT / f"results/act/phase_d/{method}/{demos}demos/{model}/train_seed{seed}"


def selected_checkpoint(run: Path) -> Path | None:
    candidates = [run / "checkpoint_selection_deterministic/validation_summary.json",
                  run / "validation_summary.json"]
    for summary in candidates:
        if summary.is_file():
            data = json.loads(summary.read_text())
            name = data.get("selected_checkpoint")
            if name is None and "selected_epoch" in data and "training_seed" in data:
                name = f"policy_epoch_{data['selected_epoch']}_seed_{data['training_seed']}.ckpt"
            path = run / name
            return path if path.is_file() else None
    return None


def candidate_checkpoints_available(run: Path) -> bool:
    manifest = run / "checkpoint_manifest.json"
    if not manifest.is_file(): return False
    try: names = [item["name"] for item in json.loads(manifest.read_text())["checkpoints"]]
    except (KeyError, TypeError, json.JSONDecodeError): return False
    return bool(names) and all((run / name).is_file() for name in names)


def requires_checkpoint_restore(demos: int, model: str, seed: int, run: Path) -> bool:
    """Return true when a completed canonical run must be restored, not retrained."""
    if seed != 0 or "phase_b_rebuild" not in run.parts:
        return False
    canonical = ROOT / f"results/act/phase_b/{demos}demos/{model}/train_seed{seed}"
    return (canonical / ".training_complete").exists() and not candidate_checkpoints_available(canonical)


def add_job(jobs, name, stage, variables, output_marker, eligible, reason, bytes_needed=0):
    partial = (output_marker.parent.is_dir() and not output_marker.exists() and
               any(output_marker.parent.iterdir()))
    jobs.append({"name": name, "stage": stage, "variables": variables,
                 "output_marker": str(output_marker.relative_to(ROOT)),
                 "status": "complete" if output_marker.exists() else
                           ("blocked" if partial else ("eligible" if eligible else "blocked")),
                 "reason": "artifact exists" if output_marker.exists() else
                           ("incomplete artifact directory requires audit" if partial else reason),
                 "estimated_bytes": bytes_needed})


def inventory(config):
    jobs = []
    seeds = config["training"]["seeds"]
    views = config["training"]["views"]
    checkpoint_bytes = config["storage"]["checkpoint_bytes_per_training_run"]
    gated_checkpoint_bytes = config["storage"]["gated_checkpoint_bytes_per_training_run"]
    eval_bytes = config["storage"]["timeseries_bytes_per_100_rollouts"]
    b0_gate = ROOT / "results/act/phase_b/preflight/phase_b0_complete.json"
    b_gate = ROOT / "results/act/phase_b/preflight/phase_b_complete.json"
    c_gate = ROOT / "results/act/phase_c/phase_c_complete.json"
    for demos in config["phase_b"]["new_training_demo_counts"]:
        for model in views:
            for seed in seeds:
                run = source_dir("phase_b", "naive", demos, model, seed)
                namespace = "phase_b_rebuild" if "phase_b_rebuild" in run.parts else "phase_b"
                variables = dict(EXPERIMENT_STAGE="phase_b", TRAINING_METHOD="naive",
                                 DEMO_COUNT=str(demos), VIEW_CONFIG=model, TRAIN_SEED=str(seed),
                                 OUTPUT_NAMESPACE=namespace)
                restore_required = requires_checkpoint_restore(demos, model, seed, run)
                train_eligible = (seed == 0 or b0_gate.exists()) and not restore_required
                add_job(jobs, f"b-train-{demos}-{model}-s{seed}", "train", variables,
                        run / ".training_complete", train_eligible,
                        "ready" if train_eligible else
                        ("completed canonical run requires checkpoint restore" if restore_required
                         else "Phase B0 audit gate missing"), checkpoint_bytes)
                selection = run / "checkpoint_selection_deterministic/.complete"
                selection_ready = (run / ".training_complete").exists() and candidate_checkpoints_available(run)
                add_job(jobs, f"b-select-{demos}-{model}-s{seed}", "select", variables | {
                    "SELECTION_NAME": "checkpoint_selection_deterministic", "DETERMINISTIC_EVAL": "1"},
                    selection, selection_ready, "ready" if selection_ready else "training incomplete")
                add_job(jobs, f"b-test-{demos}-{model}-s{seed}", "test", variables,
                        run / "eval_test/.complete", selection.exists(), "selection incomplete", eval_bytes)

    for demos in config["phase_c"]["demo_counts"]:
        for model in views:
            for seed in seeds:
                run = source_dir("phase_b", "naive", demos, model, seed)
                checkpoint_ready = selected_checkpoint(run) is not None
                condition_cameras = {"clean": ["none"]}
                for condition in config["phase_c"]["conditions"]:
                    targets = condition_cameras.get(condition, cameras(model))
                    for camera in targets:
                        label = condition if camera == "none" else f"{condition}_{camera}"
                        output = ROOT / f"results/act/phase_c/{demos}demos/{model}/train_seed{seed}/{label}/.complete"
                        variables = dict(SOURCE_STAGE="phase_b", TRAINING_METHOD="naive",
                                         DEMO_COUNT=str(demos), VIEW_CONFIG=model, TRAIN_SEED=str(seed),
                                         ROBUSTNESS_CONDITION=condition, CORRUPTION_CAMERA=camera)
                        eligible = checkpoint_ready and b_gate.exists()
                        reason = "ready" if eligible else ("Phase B completion gate missing" if checkpoint_ready else "selected checkpoint unavailable")
                        add_job(jobs, f"c-{demos}-{model}-s{seed}-{label}", "robust", variables,
                                output, eligible, reason, eval_bytes)

    for method in config["phase_d"]["methods"]:
        for demos in config["phase_d"]["demo_counts"]:
            for seed in seeds:
                model = "multi"
                run = source_dir("phase_d", method, demos, model, seed)
                variables = dict(EXPERIMENT_STAGE="phase_d", TRAINING_METHOD=method,
                                 DEMO_COUNT=str(demos), VIEW_CONFIG=model, TRAIN_SEED=str(seed))
                train_bytes = gated_checkpoint_bytes if method == "viewdrop_gated" else checkpoint_bytes
                add_job(jobs, f"d-train-{method}-{demos}-s{seed}", "train", variables,
                        run / ".training_complete", c_gate.exists(),
                        "ready" if c_gate.exists() else "Phase C completion gate missing", train_bytes)
                selection = run / "checkpoint_selection_deterministic/.complete"
                selection_ready = (run / ".training_complete").exists() and c_gate.exists()
                add_job(jobs, f"d-select-{method}-{demos}-s{seed}", "select", variables | {
                    "SELECTION_NAME": "checkpoint_selection_deterministic", "DETERMINISTIC_EVAL": "1"},
                    selection, selection_ready,
                    "ready" if selection_ready else "training or Phase C gate incomplete")
                test_ready = selection.exists() and c_gate.exists()
                add_job(jobs, f"d-test-{method}-{demos}-s{seed}", "test", variables,
                        run / "eval_test/.complete", test_ready,
                        "ready" if test_ready else "selection or Phase C gate incomplete", eval_bytes)
                checkpoint_ready = selected_checkpoint(run) is not None
                for condition in config["phase_c"]["conditions"]:
                    targets = ["none"] if condition == "clean" else ["top", "side"]
                    for camera in targets:
                        label = condition if camera == "none" else f"{condition}_{camera}"
                        robust_vars = dict(SOURCE_STAGE="phase_d", TRAINING_METHOD=method,
                                           DEMO_COUNT=str(demos), VIEW_CONFIG=model, TRAIN_SEED=str(seed),
                                           ROBUSTNESS_CONDITION=condition, CORRUPTION_CAMERA=camera)
                        robust_ready = checkpoint_ready and c_gate.exists()
                        add_job(jobs, f"d-robust-{method}-{demos}-s{seed}-{label}", "robust",
                                robust_vars, run / f"eval_robustness/{label}/.complete",
                                robust_ready, "ready" if robust_ready else
                                "selected checkpoint or Phase C gate unavailable", eval_bytes)
    return jobs


def qsub_command(job):
    script = {"train": "scripts/run_phase_b_train.pbs",
              "select": "scripts/run_phase_b_checkpoint_selection.pbs",
              "test": "scripts/run_phase_b_test.pbs",
              "robust": "scripts/run_robustness_eval.pbs"}[job["stage"]]
    variables = ",".join(f"{key}={value}" for key, value in sorted(job["variables"].items()))
    return ["qsub", "-v", variables, script]


def mark_active_pbs_jobs(jobs):
    """Prevent duplicate submissions by matching stage script and PBS variables."""
    try:
        listing = subprocess.run(["qstat", "-u", os.environ.get("USER", "")],
                                 check=True, text=True, capture_output=True)
        # Table output truncates long server suffixes with '*'; the numeric PBS
        # sequence is accepted by qstat and is stable for this server.
        job_ids = re.findall(r"(?m)^(\d+)", listing.stdout)
        if not job_ids: return 0
        result = subprocess.run(["qstat", "-f", *job_ids], check=True, text=True,
                                capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return 0
    flattened = result.stdout.replace("\n\t", "").replace("\n ", "")
    blocks = flattened.split("Job Id:")[1:]
    reserved_bytes = 0
    for job in jobs:
        if job["status"] == "complete": continue
        script = qsub_command(job)[-1]
        for block in blocks:
            if script not in block: continue
            required = {key: value for key, value in job["variables"].items()
                        if not ((key == "EXPERIMENT_STAGE" and value == "phase_b") or
                                (key == "TRAINING_METHOD" and value == "naive") or
                                (key == "OUTPUT_NAMESPACE" and value == "phase_b"))}
            if all(f"{key}={value}" in block for key, value in required.items()):
                job["status"] = "blocked"
                job["reason"] = "matching PBS job is already active"
                reserved_bytes += job["estimated_bytes"]
                break
    return reserved_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "select", "test", "robust", "all"], default="all")
    parser.add_argument("--submit", action="store_true", help="Submit eligible jobs; default is dry-run")
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--write-status", default=None)
    args = parser.parse_args()
    if args.max_jobs < 1:
        raise SystemExit("--max-jobs must be positive")
    config = json.loads(CONFIG.read_text())
    jobs = inventory(config)
    active_reserved_bytes = mark_active_pbs_jobs(jobs)
    selected = [job for job in jobs if job["status"] == "eligible" and
                (args.stage == "all" or job["stage"] == args.stage)]
    usage = shutil.disk_usage(ROOT)
    reserve = config["storage"]["safety_reserve_bytes"]
    submitted = []
    for job in selected[:args.max_jobs]:
        command = qsub_command(job)
        if args.submit:
            if (usage.free - active_reserved_bytes -
                    sum(item["estimated_bytes"] for item in submitted) <
                    job["estimated_bytes"] + reserve):
                break
            result = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
            job["job_id"] = result.stdout.strip()
            submitted.append(job)
        else:
            job["command"] = " ".join(command)
    report = {"generated_unix_seconds": time.time(), "dry_run": not args.submit,
              "disk_free_bytes": usage.free, "safety_reserve_bytes": reserve,
              "active_job_reserved_bytes": active_reserved_bytes,
              "counts": {state: sum(j["status"] == state for j in jobs)
                         for state in ("complete", "eligible", "blocked")},
              "selected": selected[:args.max_jobs], "submitted": submitted, "jobs": jobs}
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if submitted:
        registry = ROOT / "results/act/pipeline_submissions.jsonl"
        registry.parent.mkdir(parents=True, exist_ok=True)
        with registry.open("a") as stream:
            for job in submitted:
                stream.write(json.dumps({"submitted_unix_seconds": time.time(), **job}, sort_keys=True) + "\n")
    if args.write_status:
        target = Path(args.write_status)
        if not target.is_absolute(): target = ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise SystemExit(f"refusing to overwrite existing status: {target}")
        target.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
