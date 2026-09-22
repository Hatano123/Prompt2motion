#!/usr/bin/env python3
"""Audit and visualize saved Clean-50 rollouts for paired dynamic diagnosis.

This script is deliberately read-only with respect to rollout/model artifacts.  It
uses the saved JSONL summaries and MP4 files only; it does not run the simulator,
select a checkpoint, or infer unavailable per-timestep telemetry.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TARGET_CASES = ((0, 3), (1, 0), (1, 3), (2, 2), (2, 3))
MODELS = ("single_side", "multi")
CAMERAS = ("top", "side", "angle", "vis")
SAMPLE_FRAMES = (0, 80, 160, 220, 260, 300, 330, 350, 370, 390, 399)
DETAIL_FRAMES = (280, 300, 320, 340, 360, 380, 399)
MANUAL_REVIEW = {
    (0, 3): {
        "failure_type": "angular_misalignment",
        "visual_evidence": (
            "Failed multi peg is nearly level before contact, then deflects upward after "
            "contact and remains only shallowly engaged; paired side stays level and inserts."
        ),
        "confidence": "medium",
    },
    (1, 0): {
        "failure_type": "angular_misalignment",
        "visual_evidence": (
            "Failed multi reaches the socket with similar top-view alignment, but the peg "
            "tilts upward after contact; paired side remains level and inserts."
        ),
        "confidence": "high",
    },
    (1, 3): {
        "failure_type": "angular_misalignment",
        "visual_evidence": (
            "Failed multi contacts the socket edge and develops a visible vertical-angle "
            "error instead of entering; paired side approaches level and inserts."
        ),
        "confidence": "high",
    },
    (2, 2): {
        "failure_type": "angular_misalignment",
        "visual_evidence": (
            "Failed side peg is deflected upward after edge contact and stalls; paired multi "
            "maintains a level insertion direction and succeeds."
        ),
        "confidence": "high",
    },
    (2, 3): {
        "failure_type": "grasp_instability",
        "visual_evidence": (
            "Failed side loses the red peg after contact (clearly separated from the right "
            "gripper by frames 360-399); paired multi retains it and inserts."
        ),
        "confidence": "high",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def video_metadata(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    metadata = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return metadata


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {index}: {path}")
    return frame


def camera_crop(frame: np.ndarray, camera: str) -> np.ndarray:
    camera_index = CAMERAS.index(camera)
    pane_width = frame.shape[1] // len(CAMERAS)
    return frame[:, camera_index * pane_width : (camera_index + 1) * pane_width]


def labeled_tile(image: np.ndarray, label: str, width: int = 256) -> np.ndarray:
    height = round(image.shape[0] * width / image.shape[1])
    tile = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (width, 24), (0, 0, 0), -1)
    cv2.putText(tile, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def make_contact_sheet(
    paths: dict[str, Path], camera: str, output_path: Path, frame_count: int
) -> None:
    indices = [min(index, frame_count - 1) for index in SAMPLE_FRAMES]
    rows = []
    for model in MODELS:
        tiles = []
        for index in indices:
            image = camera_crop(read_frame(paths[model], index), camera)
            tiles.append(labeled_tile(image, f"{model}  t={index}"))
        rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Cannot write image: {output_path}")


def make_detail_sheet(
    paths: dict[str, Path], camera: str, output_path: Path, frame_count: int
) -> None:
    """Create a common-crop close-up of the coloured peg/socket region."""
    indices = [min(index, frame_count - 1) for index in DETAIL_FRAMES]
    images: dict[tuple[str, int], np.ndarray] = {}
    points = []
    for model in MODELS:
        for index in indices:
            image = camera_crop(read_frame(paths[model], index), camera)
            images[(model, index)] = image
            blue = (image[:, :, 0] > 120) & (image[:, :, 0] > image[:, :, 1] * 1.4)
            red = (image[:, :, 2] > 120) & (image[:, :, 2] > image[:, :, 1] * 1.4)
            ys, xs = np.where(blue | red)
            if len(xs):
                points.append((xs.min(), ys.min(), xs.max(), ys.max()))
    if not points:
        raise RuntimeError(f"Could not locate coloured objects in {paths}")
    x0 = max(0, min(point[0] for point in points) - 70)
    y0 = max(0, min(point[1] for point in points) - 70)
    x1 = min(640, max(point[2] for point in points) + 71)
    y1 = min(480, max(point[3] for point in points) + 71)
    rows = []
    for model in MODELS:
        tiles = [
            labeled_tile(images[(model, index)][y0:y1, x0:x1], f"{model}  t={index}", 320)
            for index in indices
        ]
        rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"Cannot write image: {output_path}")


def first_visual_contact_proxy(path: Path) -> int | None:
    """First 3-frame run where red/blue masks are within ~3 rendered pixels.

    This is a reproducible video proximity proxy, not the simulator's reward-3
    transition.  The ``vis`` camera is downsampled by two for the calculation.
    """
    cap = cv2.VideoCapture(str(path))
    distances: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        image = camera_crop(frame, "vis")[::2, ::2]
        blue = (
            (image[:, :, 0] > 120)
            & (image[:, :, 0] > image[:, :, 1] * 1.4)
            & (image[:, :, 0] > image[:, :, 2] * 1.2)
        )
        red = (
            (image[:, :, 2] > 120)
            & (image[:, :, 2] > image[:, :, 1] * 1.4)
            & (image[:, :, 2] > image[:, :, 0] * 1.2)
        )
        if not blue.any() or not red.any():
            distances.append(float("inf"))
            continue
        ys, xs = np.where(blue | red)
        y0, y1 = max(0, int(ys.min()) - 5), min(image.shape[0], int(ys.max()) + 6)
        x0, x1 = max(0, int(xs.min()) - 5), min(image.shape[1], int(xs.max()) + 6)
        blue_crop = blue[y0:y1, x0:x1]
        red_crop = red[y0:y1, x0:x1]
        distance = cv2.distanceTransform((~blue_crop).astype(np.uint8), cv2.DIST_L2, 3)
        distances.append(float(distance[red_crop].min()) * 2.0)
    cap.release()
    for index in range(len(distances) - 2):
        if max(distances[index : index + 3]) <= 3.0:
            return index
    return None


def classify_pattern(side: dict[str, Any], multi: dict[str, Any]) -> str:
    if side["success"] and not multi["success"]:
        return "side_success_multi_fail"
    if multi["success"] and not side["success"]:
        return "multi_success_side_fail"
    if side["success"] and multi["success"]:
        return "both_success"
    return "both_fail"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/act/clean50"))
    args = parser.parse_args()
    root = args.root
    analysis = root / "analysis"
    frames_dir = analysis / "dynamic_frames"
    analysis.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed, state in TARGET_CASES:
        records: dict[str, dict[str, Any]] = {}
        paths: dict[str, Path] = {}
        metadata: dict[str, dict[str, float | int]] = {}
        for model in MODELS:
            eval_dir = root / model / f"train_seed{seed}" / "eval_test"
            all_records = read_jsonl(eval_dir / "rollouts_selected_policy.jsonl")
            matches = [record for record in all_records if record["rollout"] == state]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one rollout for {model}, seed={seed}, state={state}")
            records[model] = matches[0]
            paths[model] = eval_dir / records[model]["video"]
            metadata[model] = video_metadata(paths[model])

        if records["single_side"]["initial_state"] != records["multi"]["initial_state"]:
            raise RuntimeError(f"Initial-state mismatch at seed={seed}, state={state}")
        if metadata["single_side"] != metadata["multi"]:
            raise RuntimeError(f"Video-metadata mismatch at seed={seed}, state={state}")
        common = metadata["single_side"]
        if common["width"] % len(CAMERAS):
            raise RuntimeError(f"Video width does not split into four cameras: {paths['single_side']}")

        for camera in CAMERAS:
            make_contact_sheet(
                paths,
                camera,
                frames_dir / f"seed{seed}_state{state}_{camera}.png",
                int(common["frame_count"]),
            )
            make_detail_sheet(
                paths,
                camera,
                frames_dir / f"seed{seed}_state{state}_{camera}_detail.png",
                int(common["frame_count"]),
            )

        pattern = classify_pattern(records["single_side"], records["multi"])
        if pattern not in {"side_success_multi_fail", "multi_success_side_fail"}:
            raise RuntimeError(f"Target is not discordant at seed={seed}, state={state}: {pattern}")
        failed_model = "multi" if pattern == "side_success_multi_fail" else "single_side"
        successful_model = "single_side" if failed_model == "multi" else "multi"
        if records[failed_model]["highest_reward"] != 3.0:
            raise RuntimeError(f"Failed member did not stop at reward 3: seed={seed}, state={state}")
        if records[successful_model]["highest_reward"] != 4.0:
            raise RuntimeError(f"Successful member did not reach reward 4: seed={seed}, state={state}")
        contact_proxy = {
            model: first_visual_contact_proxy(paths[model]) for model in MODELS
        }
        review = MANUAL_REVIEW[(seed, state)]
        rows.append({
            "seed": seed,
            "state": state,
            "paired_pattern": pattern,
            "initial_state_exact_match": True,
            "failed_model": failed_model,
            "side_episode_return": records["single_side"]["episode_return"],
            "side_highest_reward": records["single_side"]["highest_reward"],
            "side_success": records["single_side"]["success"],
            "multi_episode_return": records["multi"]["episode_return"],
            "multi_highest_reward": records["multi"]["highest_reward"],
            "multi_success": records["multi"]["success"],
            "episode_length_frames": common["frame_count"],
            "fps": common["fps"],
            "duration_seconds": float(common["frame_count"]) / float(common["fps"]),
            "side_first_visual_contact_proxy_frame": contact_proxy["single_side"],
            "side_visual_proxy_remaining_frames": (
                int(common["frame_count"]) - 1 - int(contact_proxy["single_side"])
                if contact_proxy["single_side"] is not None else "unavailable"
            ),
            "multi_first_visual_contact_proxy_frame": contact_proxy["multi"],
            "multi_visual_proxy_remaining_frames": (
                int(common["frame_count"]) - 1 - int(contact_proxy["multi"])
                if contact_proxy["multi"] is not None else "unavailable"
            ),
            "side_video": str(paths["single_side"]),
            "multi_video": str(paths["multi"]),
            # These values are not present in the JSONL or MP4 container.
            "first_reward3_timestep": "unavailable",
            "first_reward4_timestep": "unavailable",
            "reward3_remaining_steps": "unavailable",
            "return_after_reward3": "unavailable",
            "action_magnitude": "unavailable",
            "action_delta": "unavailable",
            "left_action_delta": "unavailable",
            "right_action_delta": "unavailable",
            "telemetry_oscillation_metric": "unavailable",
            "failure_type": review["failure_type"],
            "visual_evidence": review["visual_evidence"],
            "confidence": review["confidence"],
        })

    cases_path = analysis / "paired_dynamic_cases.csv"
    with cases_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    availability_rows = [
        ("episode_return", "yes", "rollout JSONL: episode_return"),
        ("highest_reward", "yes", "rollout JSONL: highest_reward"),
        ("success", "yes", "rollout JSONL: success"),
        ("initial_state", "yes", "rollout JSONL: initial_state"),
        ("episode_length", "yes", "MP4 frame count; all evaluated episodes use the full horizon"),
        ("per_timestep_reward", "no", "kept in memory during eval but not serialized"),
        ("action", "no", "kept in memory during eval but not serialized"),
        ("joint_position", "no", "kept in memory during eval but not serialized"),
        ("end_effector_pose", "no", "not serialized"),
        ("peg_socket_pose_timeseries", "no", "not serialized"),
        ("gripper_state_timeseries", "no", "not serialized"),
        ("contact_information", "no", "not serialized"),
        ("first_reward3_timestep", "no", "requires per-timestep rewards"),
        ("first_reward4_timestep", "no", "requires per-timestep rewards"),
    ]
    with (analysis / "paired_dynamic_log_availability.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("field", "available", "source_or_reason"))
        writer.writerows(availability_rows)

    failure_counts: dict[str, int] = {}
    for row in rows:
        failure_counts[row["failure_type"]] = failure_counts.get(row["failure_type"], 0) + 1
    with (analysis / "paired_dynamic_failure_types.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("failure_type", "count", "fraction_of_reviewed_cases"))
        for failure_type, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow((failure_type, count, count / len(rows)))

    print(f"Wrote {cases_path} ({len(rows)} paired cases)")
    print(f"Wrote {len(rows) * len(CAMERAS) * 2} contact/detail sheets under {frames_dir}")


if __name__ == "__main__":
    main()
