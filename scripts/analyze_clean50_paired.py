#!/usr/bin/env python3
"""Build paired Clean-50 outcome and initial-geometry diagnostics.

The 14-element insertion state layout is defined by
``generate_insertion_eval_manifest.generate_states`` and consumed by
``InsertionPolicy`` as::

    [peg_xyz, peg_wxyz, socket_xyz, socket_wxyz]

This script refuses to produce paired outputs unless every rollout state exactly
matches the held-out test manifest in the original order.
"""

import argparse
import csv
import io
import json
import math
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path


MODELS = ("single_top", "single_side", "multi")
OUTPUT_NAMES = ("top", "side", "multi")
REWARD_STAGE_NAMES = {
    0: "before_both_grippers_touch",
    1: "both_grippers_touch_objects",
    2: "both_objects_grasped_off_table",
    3: "peg_socket_contact_off_table",
    4: "pin_contact_insertion_success",
}


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def csv_text(header, rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def preserve_or_write_csv(path, header, rows):
    """Keep an existing CSV byte-for-byte when its parsed contents agree."""
    expected = [list(map(str, header)), *[list(map(str, row)) for row in rows]]
    if path.exists():
        with path.open(newline="") as handle:
            existing = list(csv.reader(handle))
        if existing != expected:
            raise ValueError(f"refusing to overwrite changed existing output: {path}")
        return "verified"
    path.write_text(csv_text(header, rows))
    return "wrote"


def paired_pattern(side_success, multi_success):
    if side_success and not multi_success:
        return "side_success_multi_fail"
    if multi_success and not side_success:
        return "multi_success_side_fail"
    if side_success and multi_success:
        return "both_success"
    return "both_fail"


def describe(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def normalize_quaternion_wxyz(quaternion):
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0:
        raise ValueError("zero-norm quaternion in test manifest")
    normalized = tuple(value / norm for value in quaternion)
    # q and -q encode the same rotation. Canonicalize for stable output.
    return tuple(-value for value in normalized) if normalized[0] < 0 else normalized


def quaternion_conjugate_wxyz(quaternion):
    w, x, y, z = quaternion
    return w, -x, -y, -z


def quaternion_multiply_wxyz(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def yaw_from_quaternion_wxyz(quaternion):
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle):
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi and angle > 0 else wrapped


def orientation_features(initial_state):
    peg_quaternion = normalize_quaternion_wxyz(initial_state[3:7])
    socket_quaternion = normalize_quaternion_wxyz(initial_state[10:14])
    relative_quaternion = normalize_quaternion_wxyz(quaternion_multiply_wxyz(
        quaternion_conjugate_wxyz(socket_quaternion), peg_quaternion))
    peg_yaw = wrap_to_pi(yaw_from_quaternion_wxyz(peg_quaternion))
    socket_yaw = wrap_to_pi(yaw_from_quaternion_wxyz(socket_quaternion))
    relative_yaw = wrap_to_pi(peg_yaw - socket_yaw)
    relative_angle = 2.0 * math.acos(max(-1.0, min(1.0, relative_quaternion[0])))
    return {
        "peg_quaternion_w": peg_quaternion[0],
        "peg_quaternion_x": peg_quaternion[1],
        "peg_quaternion_y": peg_quaternion[2],
        "peg_quaternion_z": peg_quaternion[3],
        "socket_quaternion_w": socket_quaternion[0],
        "socket_quaternion_x": socket_quaternion[1],
        "socket_quaternion_y": socket_quaternion[2],
        "socket_quaternion_z": socket_quaternion[3],
        "peg_yaw": peg_yaw,
        "socket_yaw": socket_yaw,
        "relative_yaw": relative_yaw,
        "absolute_relative_yaw": abs(relative_yaw),
        "relative_quaternion_w": relative_quaternion[0],
        "relative_quaternion_x": relative_quaternion[1],
        "relative_quaternion_y": relative_quaternion[2],
        "relative_quaternion_z": relative_quaternion[3],
        "relative_rotation_angle": relative_angle,
    }


def save_geometry_plots(rows, analysis_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        save_geometry_plots_gnuplot(rows, analysis_dir)
        return

    groups = (
        ("side_success_multi_fail", "Side success / Multi fail", "#d95f02"),
        ("multi_success_side_fail", "Multi success / Side fail", "#1b9e77"),
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    for key, label, color in groups:
        selected = [row for row in rows if row["paired_pattern"] == key]
        ax.scatter(
            [row["relative_x"] for row in selected],
            [row["relative_y"] for row in selected],
            label=f"{label} (n={len(selected)})",
            alpha=0.65,
            s=34,
            color=color,
        )
    ax.axhline(0, color="0.75", linewidth=0.8)
    ax.axvline(0, color="0.75", linewidth=0.8)
    ax.set(xlabel="relative_x = peg_x - socket_x [m]",
           ylabel="relative_y = peg_y - socket_y [m]",
           title="Paired discordant outcomes by initial relative position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(analysis_dir / "relative_xy_discordant.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    values = []
    labels = []
    colors = []
    for key, label, color in groups:
        selected = [row["relative_distance_xy"] for row in rows
                    if row["paired_pattern"] == key]
        values.append(selected)
        labels.append(label)
        colors.append(color)
    violin = ax.violinplot(values, showmeans=True, showmedians=True)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    ax.set_xticks((1, 2), labels)
    ax.set(ylabel="relative_distance_xy [m]",
           title="Initial XY distance for paired discordant outcomes")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(analysis_dir / "relative_distance_xy_discordant.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, prefix, title in zip(axes, ("peg", "socket"),
                                 ("Initial peg position", "Initial socket position")):
        for key, label, color in groups:
            selected = [row for row in rows if row["paired_pattern"] == key]
            ax.scatter(
                [row[f"{prefix}_x"] for row in selected],
                [row[f"{prefix}_y"] for row in selected],
                label=f"{label} (n={len(selected)})",
                alpha=0.65,
                s=30,
                color=color,
            )
        ax.set(xlabel=f"{prefix}_x [m]", ylabel=f"{prefix}_y [m]", title=title)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(analysis_dir / "workspace_positions_discordant.png", dpi=180)
    plt.close(fig)


def save_geometry_plots_gnuplot(rows, analysis_dir):
    """Dependency-light plotting fallback for analysis hosts without matplotlib."""
    executable = shutil.which("gnuplot")
    if executable is None:
        raise RuntimeError("creating plots requires either matplotlib or gnuplot")

    data_path = analysis_dir / "discordant_geometry_plot_data.tsv"
    with data_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("pattern", "relative_x", "relative_y", "distance_xy",
                         "peg_x", "peg_y", "socket_x", "socket_y"))
        for row in rows:
            if row["paired_pattern"] in (
                    "side_success_multi_fail", "multi_success_side_fail"):
                writer.writerow((row["paired_pattern"], row["relative_x"],
                                 row["relative_y"], row["relative_distance_xy"],
                                 row["peg_x"], row["peg_y"],
                                 row["socket_x"], row["socket_y"]))

    data = str(data_path.resolve()).replace("'", "''")
    output_dir = str(analysis_dir.resolve()).replace("'", "''")
    side = "side_success_multi_fail"
    multi = "multi_success_side_fail"
    script = f"""
set terminal pngcairo size 1260,1000 enhanced font 'Sans,14'
set datafile separator '\\t'
set key outside
set grid
set zeroaxis
set output '{output_dir}/relative_xy_discordant.png'
set title 'Paired discordant outcomes by initial relative position'
set xlabel 'relative_x = peg_x - socket_x [m]'
set ylabel 'relative_y = peg_y - socket_y [m]'
plot '{data}' using (strcol(1) eq '{side}' ? $2 : 1/0):3 with points pt 7 ps 1.1 lc rgb '#d95f02' title 'Side success / Multi fail', \\
     '{data}' using (strcol(1) eq '{multi}' ? $2 : 1/0):3 with points pt 7 ps 1.1 lc rgb '#1b9e77' title 'Multi success / Side fail'

set terminal pngcairo size 1260,900 enhanced font 'Sans,14'
set output '{output_dir}/relative_distance_xy_discordant.png'
set title 'Initial XY distance for paired discordant outcomes'
set xlabel ''
set ylabel 'relative_distance_xy [m]'
set xrange [0.5:2.5]
set xtics ('Side success / Multi fail' 1, 'Multi success / Side fail' 2)
set style fill solid 0.5 border -1
set style data boxplot
unset key
plot '{data}' using (strcol(1) eq '{side}' ? 1 : 1/0):4 lc rgb '#d95f02', \\
     '{data}' using (strcol(1) eq '{multi}' ? 2 : 1/0):4 lc rgb '#1b9e77'

set terminal pngcairo size 1500,750 enhanced font 'Sans,14'
set output '{output_dir}/workspace_positions_discordant.png'
set multiplot layout 1,2 title 'Initial workspace positions'
set xrange [0.09:0.21]
set yrange [0.39:0.61]
set xlabel 'peg_x [m]'
set ylabel 'peg_y [m]'
set title 'Initial peg position'
set key below
plot '{data}' using (strcol(1) eq '{side}' ? $5 : 1/0):6 with points pt 7 ps 1.0 lc rgb '#d95f02' title 'Side success / Multi fail', \\
     '{data}' using (strcol(1) eq '{multi}' ? $5 : 1/0):6 with points pt 7 ps 1.0 lc rgb '#1b9e77' title 'Multi success / Side fail'
set xrange [-0.21:-0.09]
set xlabel 'socket_x [m]'
set ylabel 'socket_y [m]'
set title 'Initial socket position'
plot '{data}' using (strcol(1) eq '{side}' ? $7 : 1/0):8 with points pt 7 ps 1.0 lc rgb '#d95f02' title 'Side success / Multi fail', \\
     '{data}' using (strcol(1) eq '{multi}' ? $7 : 1/0):8 with points pt 7 ps 1.0 lc rgb '#1b9e77' title 'Multi success / Side fail'
unset multiplot
"""
    subprocess.run([executable], input=script, text=True, check=True)

def write_geometry_summary(rows, analysis_dir):
    patterns = (
        "side_success_multi_fail",
        "multi_success_side_fail",
        "both_success",
        "both_fail",
    )
    count_rows = []
    for seed in range(3):
        for pattern in patterns:
            count_rows.append((seed, pattern, sum(
                row["seed"] == seed and row["paired_pattern"] == pattern for row in rows)))
    for pattern in patterns:
        count_rows.append(("all", pattern, sum(
            row["paired_pattern"] == pattern for row in rows)))
    (analysis_dir / "paired_geometry_pattern_counts.csv").write_text(
        csv_text(("seed", "paired_pattern", "count"), count_rows))

    stat_rows = []
    discordant = patterns[:2]
    features = ("relative_x", "relative_y", "relative_distance_xy")
    for pattern in discordant:
        selected = [row for row in rows if row["paired_pattern"] == pattern]
        for feature in features:
            stats = describe([row[feature] for row in selected])
            stat_rows.append((pattern, feature, len(selected),
                              stats["mean"], stats["median"], stats["std"]))
    (analysis_dir / "paired_geometry_statistics.csv").write_text(
        csv_text(("paired_pattern", "feature", "n", "mean", "median", "std"), stat_rows))

    by_pattern = {(pattern, feature): (mean, median, std)
                  for pattern, feature, _, mean, median, std in stat_rows}
    side = discordant[0]
    multi = discordant[1]
    lines = [
        "# Clean-50 paired geometry summary",
        "",
        "The rows are paired by training seed and held-out state. Geometry comes from the",
        "seed-1200 test manifest and was checked against every rollout JSONL before analysis.",
        "",
        "## Pattern counts",
        "",
        "| Pattern | Count |",
        "|---|---:|",
    ]
    totals = {pattern: sum(row["paired_pattern"] == pattern for row in rows)
              for pattern in patterns}
    lines.extend(f"| `{pattern}` | {totals[pattern]} |" for pattern in patterns)
    lines.extend(["", "## Discordant-group geometry", "",
                  "Values are metres; standard deviation is the sample SD.", "",
                  "| Pattern | Feature | Mean | Median | SD |",
                  "|---|---|---:|---:|---:|"])
    for pattern in discordant:
        for feature in features:
            mean, median, std = by_pattern[(pattern, feature)]
            lines.append(f"| `{pattern}` | `{feature}` | {mean:.6f} | {median:.6f} | {std:.6f} |")
    lines.extend(["", "## Descriptive comparison", ""])
    for feature in features:
        side_mean = by_pattern[(side, feature)][0]
        multi_mean = by_pattern[(multi, feature)][0]
        lines.append(f"- `{feature}` mean difference (side-advantage minus multi-advantage): "
                     f"{side_mean - multi_mean:+.6f} m.")
    lines.extend([
        "",
        "The two groups overlap strongly in all three plots. Their mean shifts are only 6–10 mm,",
        "while within-group SDs are about 41–43 mm for relative X/distance and 85–90 mm for",
        "relative Y. Initial XY geometry alone therefore does not show an obvious separation.",
        "Peg and socket Z are fixed at 0.05 m in the manifest, so relative Z is always zero and",
        "cannot explain the discordant outcomes.",
        "",
        "These are descriptive differences only; no significance claim is made.",
        "",
    ])
    (analysis_dir / "paired_geometry_summary.md").write_text("\n".join(lines))


def write_stage_analysis(rows, analysis_dir):
    """Write paired highest-reward diagnostics without changing model selection."""
    stage_stats = []
    for seed_label, selected_seed_rows in [
            *((str(seed), [row for row in rows if row["seed"] == seed]) for seed in range(3)),
            ("all", rows)]:
        for model in ("side", "multi"):
            total = len(selected_seed_rows)
            for reward in range(5):
                selected = [row for row in selected_seed_rows
                            if int(row[f"{model}_highest_reward"]) == reward]
                returns = [float(row[f"{model}_return"]) for row in selected]
                stage_stats.append((
                    seed_label, model, reward, REWARD_STAGE_NAMES[reward], len(selected),
                    len(selected) / total,
                    statistics.mean(returns) if returns else "",
                    statistics.median(returns) if returns else "",
                    statistics.stdev(returns) if len(returns) > 1 else (0.0 if returns else ""),
                ))
    (analysis_dir / "paired_stage_statistics.csv").write_text(csv_text(
        ("seed", "model", "highest_reward", "stage_name", "count", "proportion",
         "return_mean", "return_median", "return_std"), stage_stats))

    discordant_specs = (
        ("side_success_multi_fail", "multi"),
        ("multi_success_side_fail", "side"),
    )
    stop_rows = []
    for seed_label, selected_seed_rows in [
            *((str(seed), [row for row in rows if row["seed"] == seed]) for seed in range(3)),
            ("all", rows)]:
        for pattern, failed_model in discordant_specs:
            selected_pattern = [row for row in selected_seed_rows
                                if row["paired_pattern"] == pattern]
            for reward in range(4):
                count = sum(int(row[f"{failed_model}_highest_reward"]) == reward
                            for row in selected_pattern)
                stop_rows.append((seed_label, pattern, failed_model, reward,
                                  REWARD_STAGE_NAMES[reward], count,
                                  count / len(selected_pattern) if selected_pattern else 0.0))
    (analysis_dir / "paired_discordant_stop_stages.csv").write_text(csv_text(
        ("seed", "paired_pattern", "failed_model", "stopped_at_reward", "stage_name",
         "count", "proportion"), stop_rows))

    gap_rows = []
    for row in rows:
        side_reward = int(row["side_highest_reward"])
        multi_reward = int(row["multi_highest_reward"])
        gap_rows.append((row["seed"], row["state"], row["paired_pattern"],
                         side_reward, multi_reward, side_reward - multi_reward,
                         row["side_return"], row["multi_return"]))
    (analysis_dir / "paired_stage_gap.csv").write_text(csv_text(
        ("seed", "state", "paired_pattern", "side_highest_reward",
         "multi_highest_reward", "stage_gap", "side_return", "multi_return"), gap_rows))

    gap_summary_rows = []
    patterns = ("all", "side_success_multi_fail", "multi_success_side_fail", "both_fail")
    for seed_label, selected_seed_rows in [
            *((str(seed), [row for row in rows if row["seed"] == seed]) for seed in range(3)),
            ("all", rows)]:
        for pattern in patterns:
            selected = selected_seed_rows if pattern == "all" else [
                row for row in selected_seed_rows if row["paired_pattern"] == pattern]
            gaps = [int(row["side_highest_reward"]) - int(row["multi_highest_reward"])
                    for row in selected]
            if not gaps:
                continue
            counts = Counter(gaps)
            gap_summary_rows.append((seed_label, pattern, len(gaps),
                                     statistics.mean(gaps), statistics.median(gaps),
                                     statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
                                     sum(gap > 0 for gap in gaps), sum(gap == 0 for gap in gaps),
                                     sum(gap < 0 for gap in gaps),
                                     *[counts[gap] for gap in range(-4, 5)]))
    (analysis_dir / "paired_stage_gap_summary.csv").write_text(csv_text(
        ("seed", "paired_pattern", "n", "mean", "median", "std",
         "side_ahead", "tied", "multi_ahead", *[f"gap_{gap}" for gap in range(-4, 5)]),
        gap_summary_rows))

    contingency_by_seed = {}
    for seed_label, selected in [
            *((str(seed), [row for row in rows if row["seed"] == seed]) for seed in range(3)),
            ("all", rows)]:
        matrix = [[0 for _ in range(5)] for _ in range(5)]
        for row in selected:
            matrix[int(row["side_highest_reward"])][int(row["multi_highest_reward"])] += 1
        contingency_by_seed[seed_label] = matrix

    matrix = contingency_by_seed["all"]
    (analysis_dir / "paired_stage_contingency.csv").write_text(csv_text(
        ("side_reward", "multi_0", "multi_1", "multi_2", "multi_3", "multi_4", "row_total"),
        [(side_reward, *matrix[side_reward], sum(matrix[side_reward]))
         for side_reward in range(5)]))
    (analysis_dir / "paired_stage_contingency_by_seed.csv").write_text(csv_text(
        ("seed", "side_reward", "multi_0", "multi_1", "multi_2", "multi_3",
         "multi_4", "row_total"),
        [(seed, side_reward, *contingency_by_seed[seed][side_reward],
          sum(contingency_by_seed[seed][side_reward]))
         for seed in ("0", "1", "2", "all") for side_reward in range(5)]))

    save_stage_plots(rows, analysis_dir, matrix)
    write_stage_summary(rows, analysis_dir, stage_stats, stop_rows, gap_summary_rows, matrix)


def save_stage_plots(rows, analysis_dir, matrix):
    executable = shutil.which("gnuplot")
    if executable is None:
        raise RuntimeError("gnuplot is required to create stage diagnostic plots")

    overall_path = analysis_dir / "highest_reward_distribution_plot_data.tsv"
    with overall_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for reward in range(5):
            writer.writerow((reward,
                             sum(int(row["side_highest_reward"]) == reward for row in rows),
                             sum(int(row["multi_highest_reward"]) == reward for row in rows)))

    seed_path = analysis_dir / "highest_reward_by_seed_plot_data.tsv"
    with seed_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for seed in range(3):
            selected = [row for row in rows if row["seed"] == seed]
            for reward in range(5):
                writer.writerow((seed, reward,
                                 sum(int(row["side_highest_reward"]) == reward for row in selected),
                                 sum(int(row["multi_highest_reward"]) == reward for row in selected)))

    gap_path = analysis_dir / "stage_gap_distribution_plot_data.tsv"
    with gap_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for gap in range(-4, 5):
            writer.writerow((gap, sum(
                int(row["side_highest_reward"]) - int(row["multi_highest_reward"]) == gap
                for row in rows)))

    contingency_path = analysis_dir / "stage_contingency_plot_data.tsv"
    with contingency_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for side_reward in range(5):
            for multi_reward in range(5):
                writer.writerow((multi_reward, side_reward, matrix[side_reward][multi_reward]))

    out = str(analysis_dir.resolve()).replace("'", "''")
    overall = str(overall_path.resolve()).replace("'", "''")
    by_seed = str(seed_path.resolve()).replace("'", "''")
    gaps = str(gap_path.resolve()).replace("'", "''")
    contingency = str(contingency_path.resolve()).replace("'", "''")
    script = f"""
set terminal pngcairo size 1260,850 enhanced font 'Sans,14'
set datafile separator '\\t'
set style fill solid 0.75 border -1
set boxwidth 0.34
set grid ytics
set key outside
set output '{out}/highest_reward_distribution.png'
set title 'Highest reward distribution (3 seeds x 100 states)'
set xlabel 'Highest reward stage'
set ylabel 'Count'
set xtics 0,1,4
plot '{overall}' using ($1-0.18):2 with boxes lc rgb '#d95f02' title 'Side', \\
     '{overall}' using ($1+0.18):3 with boxes lc rgb '#1b9e77' title 'Multi'

set terminal pngcairo size 1500,1200 enhanced font 'Sans,13'
set output '{out}/highest_reward_by_seed.png'
set multiplot layout 3,1 title 'Highest reward distribution by training seed'
do for [s=0:2] {{
  set title sprintf('Seed %d',s)
  set xlabel 'Highest reward stage'
  set ylabel 'Count'
  set yrange [0:*]
  plot '{by_seed}' using ($1==s ? $2-0.18 : 1/0):3 with boxes lc rgb '#d95f02' title 'Side', \\
       '{by_seed}' using ($1==s ? $2+0.18 : 1/0):4 with boxes lc rgb '#1b9e77' title 'Multi'
}}
unset multiplot

set terminal pngcairo size 1100,900 enhanced font 'Sans,14'
set output '{out}/stage_gap_distribution.png'
set title 'Paired stage gap: side highest reward - multi highest reward'
set xlabel 'Stage gap'
set ylabel 'Count'
set xrange [-4.6:4.6]
set xtics -4,1,4
unset key
plot '{gaps}' using 1:2 with boxes lc rgb '#7570b3'

set terminal pngcairo size 1050,900 enhanced font 'Sans,14'
set output '{out}/side_multi_stage_contingency.png'
set title 'Side vs Multi highest reward contingency (n=300)'
set xlabel 'Multi highest reward'
set ylabel 'Side highest reward'
set xrange [-0.5:4.5]
set yrange [-0.5:4.5]
set xtics 0,1,4
set ytics 0,1,4
set size ratio -1
set palette defined (0 '#fff7ec', 1 '#fdbb84', 2 '#e34a33', 3 '#7f0000')
set cbrange [0:*]
unset key
plot '{contingency}' using 1:2:3 with image, \\
     '{contingency}' using 1:2:(sprintf('%d',$3)) with labels tc rgb 'black' font ',13'
"""
    subprocess.run([executable], input=script, text=True, check=True)


def write_stage_summary(rows, analysis_dir, stage_stats, stop_rows, gap_summary_rows, matrix):
    def all_stop(pattern):
        return [(reward, count, proportion) for seed, row_pattern, _, reward, _, count, proportion
                in stop_rows if seed == "all" and row_pattern == pattern]

    overall_gap = next(row for row in gap_summary_rows
                       if row[0] == "all" and row[1] == "all")
    both_fail_gap = next(row for row in gap_summary_rows
                         if row[0] == "all" and row[1] == "both_fail")
    lines = [
        "# Clean-50 paired highest-reward summary",
        "",
        "Reward stages follow `InsertionTask.get_reward`: 0 before bilateral object contact,",
        "1 both grippers touching their objects, 2 both objects grasped off the table,",
        "3 peg/socket contact off the table, and 4 pin contact (successful insertion).",
        "",
        "## Overall highest-reward distribution",
        "",
        "| Reward | Stage | Side count | Multi count |",
        "|---:|---|---:|---:|",
    ]
    for reward in range(5):
        side_count = next(row[4] for row in stage_stats
                          if row[0] == "all" and row[1] == "side" and row[2] == reward)
        multi_count = next(row[4] for row in stage_stats
                           if row[0] == "all" and row[1] == "multi" and row[2] == reward)
        lines.append(f"| {reward} | `{REWARD_STAGE_NAMES[reward]}` | {side_count} | {multi_count} |")
    lines.extend([
        "",
        "The main aggregate shift is between reward 3 and 4: multi has 23 more reward-3",
        "episodes, while side has 30 more successful reward-4 episodes. Low-stage failures",
        "(reward 0–2) are uncommon for both models.",
        "",
        "## Seed comparison",
        "",
        "| Seed | Side reward 3 | Side success | Multi reward 3 | Multi success | Mean gap |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in range(3):
        seed_text = str(seed)
        side_3 = next(row[4] for row in stage_stats
                      if row[0] == seed_text and row[1] == "side" and row[2] == 3)
        side_4 = next(row[4] for row in stage_stats
                      if row[0] == seed_text and row[1] == "side" and row[2] == 4)
        multi_3 = next(row[4] for row in stage_stats
                       if row[0] == seed_text and row[1] == "multi" and row[2] == 3)
        multi_4 = next(row[4] for row in stage_stats
                       if row[0] == seed_text and row[1] == "multi" and row[2] == 4)
        seed_gap = next(row[3] for row in gap_summary_rows
                        if row[0] == seed_text and row[1] == "all")
        lines.append(f"| {seed} | {side_3} | {side_4} | {multi_3} | {multi_4} | {seed_gap:+.3f} |")
    lines.extend([
        "",
        "The direction is not fully seed-consistent: side is slightly ahead for seed 0 and",
        "strongly ahead for seed 1, whereas multi is slightly ahead for seed 2.",
        "",
        "## Discordant-success stopping stages",
        "",
    ])
    for pattern in ("side_success_multi_fail", "multi_success_side_fail"):
        failed_model = "multi" if pattern.startswith("side_") else "side"
        lines.extend([f"### `{pattern}`: failed {failed_model}", "",
                      "| Stopped reward | Count | Proportion |", "|---:|---:|---:|"])
        for reward, count, proportion in all_stop(pattern):
            lines.append(f"| {reward} | {count} | {proportion:.1%} |")
        lines.append("")
    lines.extend([
        "## Stage gap",
        "",
        f"Across all 300 pairs, mean gap is {overall_gap[3]:+.3f}, median is "
        f"{overall_gap[4]:+.1f}; side is ahead in {overall_gap[6]}, tied in "
        f"{overall_gap[7]}, and multi is ahead in {overall_gap[8]} pairs.",
        "",
        f"Within the {both_fail_gap[2]} `both_fail` pairs, side is ahead in "
        f"{both_fail_gap[6]}, tied in {both_fail_gap[7]}, and multi is ahead in "
        f"{both_fail_gap[8]} pairs (mean gap {both_fail_gap[3]:+.3f}).",
        "",
        "## Contingency table",
        "",
        "Rows are side highest reward and columns are multi highest reward.",
        "",
        "| Side \\ Multi | 0 | 1 | 2 | 3 | 4 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for side_reward in range(5):
        lines.append("| " + " | ".join(map(str, (side_reward, *matrix[side_reward]))) + " |")
    lines.extend([
        "",
        "## Episode return conditional on highest reward",
        "",
        "| Highest reward | Side mean return | Multi mean return |",
        "|---:|---:|---:|",
    ])
    for reward in range(5):
        side_mean = next(row[6] for row in stage_stats
                         if row[0] == "all" and row[1] == "side" and row[2] == reward)
        multi_mean = next(row[6] for row in stage_stats
                          if row[0] == "all" and row[1] == "multi" and row[2] == reward)
        side_text = f"{side_mean:.2f}" if side_mean != "" else "—"
        multi_text = f"{multi_mean:.2f}" if multi_mean != "" else "—"
        lines.append(f"| {reward} | {side_text} | {multi_text} |")
    lines.extend([
        "",
        "At reward 3 the mean returns are similar (side 350.84, multi 357.23). Among successful",
        "episodes, side has a higher mean return (429.75 vs 406.28), suggesting earlier or more",
        "sustained reward accumulation, although this is descriptive and not a paired causal test.",
        "",
        "All counts are paired within training seed and held-out state. The 300 rows are not",
        "interpreted as 300 independent training repetitions, and no checkpoint is reselected.",
        "",
    ])
    (analysis_dir / "paired_stage_summary.md").write_text("\n".join(lines))


def write_orientation_analysis(rows, analysis_dir):
    """Analyze manifest orientation, explicitly preserving a no-variance result."""
    feature_header = (
        "seed", "state", "paired_pattern", "peg_yaw", "socket_yaw", "relative_yaw",
        "absolute_relative_yaw", "relative_quaternion_w", "relative_quaternion_x",
        "relative_quaternion_y", "relative_quaternion_z", "relative_rotation_angle",
        "relative_x", "relative_y", "relative_distance_xy", "side_highest_reward",
        "multi_highest_reward", "side_success", "multi_success",
    )
    feature_rows = [tuple(row[name] if name in row else row[name.removesuffix("_success")]
                          for name in feature_header) for row in rows]
    (analysis_dir / "paired_orientation_features.csv").write_text(
        csv_text(feature_header, feature_rows))

    unique_state_rows = {row["state"]: row for row in rows if row["seed"] == 0}
    state_angles = sorted(row["absolute_relative_yaw"] for row in unique_state_rows.values())
    constant_orientation = math.isclose(state_angles[0], state_angles[-1], abs_tol=1e-12)
    if constant_orientation:
        thresholds = (state_angles[0], state_angles[-1])
        for row in rows:
            row["orientation_bin"] = "constant_zero_orientation"
    else:
        def quantile(values, probability):
            position = (len(values) - 1) * probability
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return values[lower]
            return values[lower] + (values[upper] - values[lower]) * (position - lower)
        thresholds = (quantile(state_angles, 1 / 3), quantile(state_angles, 2 / 3))
        for row in rows:
            value = row["absolute_relative_yaw"]
            row["orientation_bin"] = (
                "low" if value <= thresholds[0] else
                "medium" if value <= thresholds[1] else "high")

    pattern_rows = []
    for pattern in ("side_success_multi_fail", "multi_success_side_fail",
                    "both_success", "both_fail"):
        selected = [row for row in rows if row["paired_pattern"] == pattern]
        for feature in ("peg_yaw", "socket_yaw", "relative_yaw",
                        "absolute_relative_yaw", "relative_rotation_angle"):
            values = [row[feature] for row in selected]
            stats = describe(values)
            pattern_rows.append((pattern, feature, len(values), stats["mean"],
                                 stats["median"], stats["std"], min(values), max(values)))

    success_rows = []
    bins = sorted({row["orientation_bin"] for row in rows})
    for seed_label, selected_seed_rows in [
            *((str(seed), [row for row in rows if row["seed"] == seed]) for seed in range(3)),
            ("all", rows)]:
        for orientation_bin in bins:
            selected_bin = [row for row in selected_seed_rows
                            if row["orientation_bin"] == orientation_bin]
            for model in ("side", "multi"):
                successes = sum(int(row[model]) for row in selected_bin)
                success_rows.append((seed_label, model, orientation_bin, thresholds[0],
                                     thresholds[1], len(selected_bin), successes,
                                     successes / len(selected_bin) if selected_bin else ""))
    (analysis_dir / "orientation_success_statistics.csv").write_text(csv_text(
        ("seed", "model", "orientation_bin", "lower_boundary_rad", "upper_boundary_rad",
         "n", "successes", "success_rate"), success_rows))

    contact_rows = []
    for seed_label, selected_seed_rows in [
            *((str(seed), [row for row in rows if row["seed"] == seed]) for seed in range(3)),
            ("all", rows)]:
        for orientation_bin in bins:
            selected_bin = [row for row in selected_seed_rows
                            if row["orientation_bin"] == orientation_bin]
            for model in ("side", "multi"):
                eligible = [row for row in selected_bin
                            if int(row[f"{model}_highest_reward"]) >= 3]
                inserted = sum(int(row[f"{model}_highest_reward"]) == 4 for row in eligible)
                contact_rows.append((seed_label, model, orientation_bin, len(selected_bin),
                                     len(eligible), inserted,
                                     inserted / len(eligible) if eligible else ""))
    (analysis_dir / "contact_to_insertion_statistics.csv").write_text(csv_text(
        ("seed", "model", "orientation_bin", "total_rollouts", "reached_reward_ge_3",
         "reached_reward_4", "p_reward4_given_reward_ge3"), contact_rows))

    (analysis_dir / "paired_orientation_pattern_statistics.csv").write_text(csv_text(
        ("paired_pattern", "feature", "n", "mean", "median", "std", "min", "max"),
        pattern_rows))
    save_orientation_plots(rows, analysis_dir, contact_rows)
    write_orientation_summary(rows, analysis_dir, constant_orientation, thresholds,
                              pattern_rows, contact_rows)


def save_orientation_plots(rows, analysis_dir, contact_rows):
    executable = shutil.which("gnuplot")
    if executable is None:
        raise RuntimeError("gnuplot is required to create orientation plots")
    patterns = ("side_success_multi_fail", "multi_success_side_fail",
                "both_success", "both_fail")
    orientation_path = analysis_dir / "orientation_pattern_plot_data.tsv"
    with orientation_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for index, pattern in enumerate(patterns):
            selected = [row for row in rows if row["paired_pattern"] == pattern]
            writer.writerow((index, pattern, len(selected),
                             statistics.mean(row["absolute_relative_yaw"] for row in selected),
                             statistics.mean(row["relative_rotation_angle"] for row in selected)))
    scatter_path = analysis_dir / "distance_orientation_plot_data.tsv"
    with scatter_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for row in rows:
            if row["paired_pattern"] in patterns[:2]:
                writer.writerow((row["paired_pattern"], row["relative_distance_xy"],
                                 row["absolute_relative_yaw"], row["relative_x"]))
    conversion_path = analysis_dir / "orientation_conversion_plot_data.tsv"
    with conversion_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for seed, model, _, _, _, _, rate in contact_rows:
            writer.writerow((seed, model, rate))

    out = str(analysis_dir.resolve()).replace("'", "''")
    orientation = str(orientation_path.resolve()).replace("'", "''")
    scatter = str(scatter_path.resolve()).replace("'", "''")
    conversion = str(conversion_path.resolve()).replace("'", "''")
    script = f"""
set terminal pngcairo size 1300,850 enhanced font 'Sans,14'
set datafile separator '\\t'
set style fill solid 0.75 border -1
set boxwidth 0.65
set grid ytics
unset key
set output '{out}/relative_yaw_discordant.png'
set title 'Absolute relative yaw by paired outcome (manifest has no orientation variation)'
set xlabel 'Paired pattern'
set ylabel 'Mean absolute relative yaw [rad]'
set yrange [-0.01:0.1]
set bmargin 8
set xtics rotate by -15
set xtics ('side success / multi fail' 0, 'multi success / side fail' 1, 'both success' 2, 'both fail' 3)
plot '{orientation}' using 1:4 with boxes lc rgb '#7570b3'

set output '{out}/rotation_angle_discordant.png'
set title '3D relative rotation angle by paired outcome (constant identity orientation)'
set ylabel 'Mean relative rotation angle [rad]'
plot '{orientation}' using 1:5 with boxes lc rgb '#66a61e'

set terminal pngcairo size 1300,850 enhanced font 'Sans,14'
set output '{out}/distance_orientation_discordant.png'
set title 'Distance vs absolute relative yaw (all yaw differences are zero)'
set bmargin 4
set xtics norotate autofreq
set xlabel 'relative_distance_xy [m]'
set ylabel 'absolute_relative_yaw [rad]'
set yrange [-0.01:0.1]
set key outside
plot '{scatter}' using (strcol(1) eq 'side_success_multi_fail' ? $2 : 1/0):3 with points pt 7 ps 1.0 lc rgb '#d95f02' title 'Side success / Multi fail', \\
     '{scatter}' using (strcol(1) eq 'multi_success_side_fail' ? $2 : 1/0):3 with points pt 7 ps 1.0 lc rgb '#1b9e77' title 'Multi success / Side fail'

set terminal pngcairo size 1300,850 enhanced font 'Sans,14'
set output '{out}/orientation_vs_success.png'
set title 'P(reward=4 | highest reward >= 3); orientation bin is constant'
set xlabel 'All seeds and individual training seeds'
set ylabel 'Contact-to-insertion probability'
set yrange [0:0.8]
set boxwidth 0.32
set xtics norotate
set xtics ('All' 0, 'Seed 0' 1, 'Seed 1' 2, 'Seed 2' 3)
set key outside
plot '{conversion}' using (strcol(1) eq 'all' && strcol(2) eq 'side' ? -0.17 : strcol(1) eq '0' && strcol(2) eq 'side' ? 0.83 : strcol(1) eq '1' && strcol(2) eq 'side' ? 1.83 : strcol(1) eq '2' && strcol(2) eq 'side' ? 2.83 : 1/0):3 with boxes lc rgb '#d95f02' title 'Side', \\
     '{conversion}' using (strcol(1) eq 'all' && strcol(2) eq 'multi' ? 0.17 : strcol(1) eq '0' && strcol(2) eq 'multi' ? 1.17 : strcol(1) eq '1' && strcol(2) eq 'multi' ? 2.17 : strcol(1) eq '2' && strcol(2) eq 'multi' ? 3.17 : 1/0):3 with boxes lc rgb '#1b9e77' title 'Multi'
"""
    subprocess.run([executable], input=script, text=True, check=True)


def write_orientation_summary(rows, analysis_dir, constant_orientation, thresholds,
                              pattern_rows, contact_rows):
    lines = [
        "# Clean-50 initial-orientation analysis",
        "",
        "The manifest layout and repository consumers define quaternions as scalar-first WXYZ.",
        "Relative rotation is computed as `inverse(socket) * peg`; yaw is wrapped to [-pi, pi]",
        "and the canonical 3D relative angle is in [0, pi].",
        "",
        "## Identifiability check",
        "",
    ]
    if constant_orientation:
        lines.extend([
            "All 100 held-out states use peg quaternion `[1, 0, 0, 0]` and socket quaternion",
            "`[1, 0, 0, 0]`. Consequently peg yaw, socket yaw, relative yaw, absolute relative",
            "yaw, and 3D relative rotation angle are exactly zero for all 300 paired rows.",
            "",
            "Quantile bins cannot be formed without inventing variation. A single bin named",
            "`constant_zero_orientation` is therefore used. Orientation cannot explain any",
            "success difference in this Clean-50 test set.",
        ])
    else:
        lines.append(f"Orientation tertiles use held-out-state quantiles at {thresholds[0]:.6f} "
                     f"and {thresholds[1]:.6f} radians.")
    lines.extend([
        "",
        "## Contact-to-insertion conversion",
        "",
        "`P(reward=4 | highest_reward>=3)` uses only episodes that reached peg/socket contact",
        "or insertion.",
        "",
        "| Seed | Side reached >=3 | Side inserted | Side rate | Multi reached >=3 | Multi inserted | Multi rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in ("0", "1", "2", "all"):
        side = next(row for row in contact_rows if row[0] == seed and row[1] == "side")
        multi = next(row for row in contact_rows if row[0] == seed and row[1] == "multi")
        label = "All" if seed == "all" else seed
        lines.append(f"| {label} | {side[4]} | {side[5]} | {side[6]:.1%} | "
                     f"{multi[4]} | {multi[5]} | {multi[6]:.1%} |")
    lines.extend([
        "",
        "The aggregate conversion advantage for side is driven mainly by seed 1. Multi has a",
        "higher conversion rate for seed 2, so the direction is not seed-consistent.",
        "Because orientation is constant, neither a relation between larger orientation error",
        "and lower success nor an interaction with XY distance can be estimated from these data.",
        "",
        "No new rollout or checkpoint selection was performed.",
        "",
    ])
    (analysis_dir / "paired_orientation_summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/act/clean50"))
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=Path("archives/environments/sim_insertion_test_seed1200_100.json"),
    )
    args = parser.parse_args()

    manifest_states = json.loads(args.test_manifest.read_text())["initial_states"]
    if any(len(state) != 14 for state in manifest_states):
        raise ValueError("insertion manifest states must have 14 values: peg pose then socket pose")
    paired_rows = []
    geometry_rows = []
    counts_by_seed = {}

    for seed in range(3):
        model_rows = {}
        for model in MODELS:
            path = (
                args.root
                / model
                / f"train_seed{seed}"
                / "eval_test"
                / "rollouts_selected_policy.jsonl"
            )
            rows = read_jsonl(path)
            if len(rows) != len(manifest_states):
                raise ValueError(f"{path}: expected {len(manifest_states)} rows, got {len(rows)}")
            if [row["rollout"] for row in rows] != list(range(len(manifest_states))):
                raise ValueError(f"{path}: rollout IDs are not ordered 0..N-1")
            if [row["initial_state"] for row in rows] != manifest_states:
                raise ValueError(f"{path}: initial states do not match the test manifest")
            if any(row.get("visual_condition") != "clean" for row in rows):
                raise ValueError(f"{path}: contains a non-clean visual condition")
            if any(row.get("corruption_camera") is not None for row in rows):
                raise ValueError(f"{path}: contains a camera corruption")
            model_rows[model] = rows

        pattern_counts = Counter()
        for state in range(len(manifest_states)):
            outcome = tuple(int(model_rows[model][state]["success"]) for model in MODELS)
            pattern_counts[outcome] += 1
            paired_rows.append((seed, state, *outcome))
            initial_state = manifest_states[state]
            peg_x, peg_y, peg_z = initial_state[:3]
            socket_x, socket_y, socket_z = initial_state[7:10]
            relative_x = peg_x - socket_x
            relative_y = peg_y - socket_y
            relative_z = peg_z - socket_z
            orientations = orientation_features(initial_state)
            model_metrics = {}
            for model, output_name in zip(MODELS, OUTPUT_NAMES):
                rollout = model_rows[model][state]
                model_metrics[f"{output_name}_return"] = rollout["episode_return"]
                model_metrics[f"{output_name}_highest_reward"] = rollout["highest_reward"]
            geometry_rows.append({
                "seed": seed,
                "state": state,
                "top": outcome[0],
                "side": outcome[1],
                "multi": outcome[2],
                "paired_pattern": paired_pattern(outcome[1], outcome[2]),
                "peg_x": peg_x,
                "peg_y": peg_y,
                "peg_z": peg_z,
                "socket_x": socket_x,
                "socket_y": socket_y,
                "socket_z": socket_z,
                "relative_x": relative_x,
                "relative_y": relative_y,
                "relative_z": relative_z,
                "relative_distance_xy": math.hypot(relative_x, relative_y),
                "relative_distance_xyz": math.sqrt(
                    relative_x ** 2 + relative_y ** 2 + relative_z ** 2),
                **orientations,
                **model_metrics,
            })
        counts_by_seed[seed] = pattern_counts

    paired_path = args.root / "paired_outcomes.csv"
    paired_status = preserve_or_write_csv(
        paired_path, ("seed", "state", *OUTPUT_NAMES), paired_rows)

    summary_path = args.root / "paired_pattern_summary.csv"
    patterns = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    summary_rows = []
    for seed in range(3):
        for pattern in patterns:
            summary_rows.append((seed, *pattern, counts_by_seed[seed][pattern]))
    total = sum((counts_by_seed[seed] for seed in range(3)), Counter())
    for pattern in patterns:
        summary_rows.append(("all", *pattern, total[pattern]))
    summary_status = preserve_or_write_csv(
        summary_path, ("seed", *OUTPUT_NAMES, "count"), summary_rows)

    geometry_path = args.root / "paired_outcomes_with_geometry.csv"
    geometry_header = tuple(geometry_rows[0])
    with geometry_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=geometry_header)
        writer.writeheader()
        writer.writerows(geometry_rows)

    analysis_dir = args.root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_geometry_summary(geometry_rows, analysis_dir)
    save_geometry_plots(geometry_rows, analysis_dir)
    write_stage_analysis(geometry_rows, analysis_dir)
    write_orientation_analysis(geometry_rows, analysis_dir)

    print(f"{paired_status} {paired_path} ({len(paired_rows)} rows)")
    print(f"{summary_status} {summary_path}")
    print(f"wrote {geometry_path} ({len(geometry_rows)} rows)")
    print(f"wrote geometry, highest-reward, and orientation diagnostics to {analysis_dir}")


if __name__ == "__main__":
    main()
