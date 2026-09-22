#!/usr/bin/env python3
"""Audit Clean-50's budget and create the Phase B seed-0 preflight manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path


VIEWS = {
    "single_top": ["top"],
    "single_side": ["side"],
    "multi": ["top", "side"],
}
DEMO_COUNTS = (5, 10, 25)
BATCH_SIZE = 8
EPISODE_LENGTH = 400
BASELINE_EPOCHS = 2000
BASELINE_STEPS_PER_EPOCH = 5
TARGET_OPTIMIZER_STEPS = BASELINE_EPOCHS * BASELINE_STEPS_PER_EPOCH
VALIDATION_EVERY_STEPS = BASELINE_STEPS_PER_EPOCH
CHECKPOINT_EVERY_STEPS = 500 * BASELINE_STEPS_PER_EPOCH
CHUNK_SIZE = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_baseline(root: Path) -> dict:
    view_audits = {}
    rounded_parameter_counts = set()
    for model, cameras in VIEWS.items():
        run = root / "results" / "act" / "clean50" / model / "train_seed0"
        split = json.loads((run / "split_manifest.json").read_text())
        metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        log = (run / "train.log").read_text(errors="replace")
        parameter_match = re.search(r"number of parameters: ([0-9.]+)M", log)
        if not parameter_match:
            raise RuntimeError(f"Parameter count missing from {run / 'train.log'}")
        rounded_parameter_counts.add(parameter_match.group(1))
        if split["train_episode_ids"] != list(range(40)):
            raise RuntimeError(f"Unexpected baseline train split for {model}")
        if split["val_episode_ids"] != list(range(40, 50)):
            raise RuntimeError(f"Unexpected baseline validation split for {model}")
        if split["camera_names"] != cameras:
            raise RuntimeError(f"Unexpected baseline camera order for {model}")
        if len(metrics) != BASELINE_EPOCHS or [metrics[0]["epoch"], metrics[-1]["epoch"]] != [0, 1999]:
            raise RuntimeError(f"Baseline did not complete 2000 epochs for {model}")
        view_audits[model] = {
            "camera_names": cameras,
            "completed_epochs_from_metrics": len(metrics),
            "first_epoch": metrics[0]["epoch"],
            "last_epoch": metrics[-1]["epoch"],
            "parameter_count_log_rounded": f"{parameter_match.group(1)}M",
            "train_episode_ids": split["train_episode_ids"],
            "validation_episode_ids": split["val_episode_ids"],
        }
    if len(rounded_parameter_counts) != 1:
        raise RuntimeError("Baseline parameter count differs across views")
    return {
        "views": view_audits,
        "dataset_episodes": 40,
        "samples_per_episode_stored": EPISODE_LENGTH,
        "stored_training_transitions": 40 * EPISODE_LENGTH,
        "dataset_items_per_epoch": 40,
        "batch_size": BATCH_SIZE,
        "batch_sizes_per_epoch": [8, 8, 8, 8, 8],
        "steps_per_epoch": BASELINE_STEPS_PER_EPOCH,
        "epochs": BASELINE_EPOCHS,
        "optimizer_steps": TARGET_OPTIMIZER_STEPS,
        "optimizer_step_evidence": (
            "2000 metrics rows (epochs 0-1999) x DataLoader length 5; training code calls "
            "optimizer.step once per batch with no gradient accumulation"
        ),
        "gradient_accumulation_steps": 1,
        "sampler": "RandomSampler via shuffle=True; no replacement within each epoch",
        "shuffle": True,
        "drop_last": False,
        "validation_interval_optimizer_steps": 5,
        "validation_passes": 2000,
        "checkpoint_interval_optimizer_steps": 2500,
        "checkpoint_steps": [2500, 5000, 7500, 10000],
        "scheduler": None,
        "parameter_count_log_rounded": f"{next(iter(rounded_parameter_counts))}M",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--parameter-count", type=int,
                        help="Exact runtime parameter count obtained from the smoke test")
    args = parser.parse_args()
    root = args.root.resolve()
    archive = root / "archives/datasets/act_sim_insertion_scripted_top_side_seed0_50episodes.zip"
    output = root / "results/act/phase_b/preflight"
    output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as source:
        metadata = json.loads(source.read("sim_insertion_scripted/dataset_metadata.json"))
        episode_files = sorted(name for name in source.namelist()
                               if re.search(r"/episode_[0-9]+\.hdf5$", name))
    if metadata["camera_names"] != ["top", "side"] or len(episode_files) != 50:
        raise RuntimeError("Clean-50 archive camera order or episode count is unexpected")

    baseline = audit_baseline(root)
    runs = []
    for demos in DEMO_COUNTS:
        sampled_examples = TARGET_OPTIMIZER_STEPS * BATCH_SIZE
        exposure = sampled_examples / demos
        for model, cameras in VIEWS.items():
            runs.append({
                "run_name": f"phase_b_{demos}demos_{model}_seed0",
                "view": model,
                "demo_count": demos,
                "train_episode_ids": list(range(demos)),
                "training_seed": 0,
                "batch_size": BATCH_SIZE,
                "effective_batch_size_every_step": BATCH_SIZE,
                "optimizer": "AdamW",
                "weight_decay": 1e-4,
                "learning_rate": 1e-5,
                "backbone_learning_rate": 1e-5,
                "scheduler": None,
                "chunk_size": CHUNK_SIZE,
                "kl_weight": 10,
                "target_optimizer_steps": TARGET_OPTIMIZER_STEPS,
                "baseline_epoch_equivalent": TARGET_OPTIMIZER_STEPS / VALIDATION_EVERY_STEPS,
                "native_epoch_equivalent": exposure,
                "sampled_episode_starts": sampled_examples,
                "nominal_action_target_slots": sampled_examples * CHUNK_SIZE,
                "approximate_exposures_per_episode": exposure,
                "validation_episode_ids": list(range(40, 50)),
                "validation_every_optimizer_steps": VALIDATION_EVERY_STEPS,
                "checkpoint_optimizer_steps": [2500, 5000, 7500, 10000],
                "camera_order": cameras,
                "sampler": "CyclingRandomSampler: concatenated shuffled no-replacement passes",
                "shuffle": True,
                "drop_last": False,
                "output_directory": f"results/act/phase_b/{demos}demos/{model}/train_seed0",
                "parameter_count": args.parameter_count,
                "parameter_count_reference_rounded": baseline["parameter_count_log_rounded"],
                "architecture": "ACT ResNet18, encoder 4, decoder 7, hidden 512, FFN 3200, heads 8",
                "image_resolution": [480, 640],
                "image_preprocessing": "uint8/255 then ImageNet mean/std normalization",
                "dataset_archive": str(archive.relative_to(root)),
                "dataset_archive_sha256": sha256(archive),
            })

    manifest = {
        "experiment": "phase_b_data_amount_x_view_seed0",
        "status": "preflight",
        "baseline_40demo": baseline,
        "dataset": {
            "archive": str(archive.relative_to(root)),
            "sha256": sha256(archive),
            "num_episodes": metadata["num_episodes"],
            "camera_names": metadata["camera_names"],
            "generation_seed": metadata["generation_seed"],
            "episode_length": EPISODE_LENGTH,
        },
        "target_optimizer_steps": TARGET_OPTIMIZER_STEPS,
        "training_seed": 0,
        "runs": runs,
        "baseline_reuse": "Clean-50 40-demo seed-0 results; no retraining",
        "heldout_test": "not run in this phase",
    }
    (output / "phase_b_seed0_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    fields = list(runs[0])
    with (output / "phase_b_seed0_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in run.items()})

    lines = [
        "# Phase B seed 0 preflight audit", "",
        "## 40-demo baseline budget", "",
        f"- Stored training transitions: {baseline['stored_training_transitions']:,} "
        f"(40 episodes × {EPISODE_LENGTH})",
        "- Dataset items per epoch: 40 (one randomly selected start timestep per episode)",
        "- Batch size: 8; DataLoader length: 5; drop_last: false",
        "- Epochs: 2,000; optimizer steps: **10,000**; gradient accumulation: none",
        "- Validation: every 5 optimizer steps; checkpoint candidates: 2,500/5,000/7,500/10,000",
        "- Optimizer: AdamW, lr 1e-5, backbone lr 1e-5, weight decay 1e-4; scheduler: none",
        f"- Parameter count in all three existing logs: {baseline['parameter_count_log_rounded']}",
        "", "The old metrics did not serialize an optimizer-step counter. The 10,000 value is "
        "reconstructed from 2,000 completed metric epochs and the audited one-step-per-batch loop.",
        "", "## Phase B controlled budget", "",
        "All new runs use 10,000 optimizer steps, full effective batch size 8, and therefore "
        "80,000 sampled episode-start examples. The cycling sampler concatenates independently "
        "shuffled no-replacement passes over the nested subset.", "",
        "| demos | sampled examples | exposure/episode | baseline-equivalent epochs |",
        "|---:|---:|---:|---:|",
    ]
    if args.parameter_count is not None:
        lines.insert(10, f"- Exact runtime parameter count verified by smoke test: {args.parameter_count:,}")
    for demos in DEMO_COUNTS:
        lines.append(f"| {demos} | 80,000 | {80000 / demos:,.0f} | 2,000 |")
    lines += ["", "Training episodes are nested: 0–4, 0–9, 0–24; validation is always 40–49.",
              "The held-out test is not executed by the Phase B training jobs.", ""]
    (output / "phase_b_budget_audit.md").write_text("\n".join(lines))
    print(f"verified baseline optimizer steps: {TARGET_OPTIMIZER_STEPS}")
    print(f"wrote {len(runs)} runs under {output}")


if __name__ == "__main__":
    main()
