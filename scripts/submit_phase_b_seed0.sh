#!/bin/bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

smoke_result="results/act/phase_b_smoke/5demos/single_top/train_seed0"
manifest="results/act/phase_b/preflight/phase_b_seed0_manifest.json"
registry="results/act/phase_b/submitted_jobs.tsv"

[[ -e "$smoke_result/.training_complete" ]] || {
  echo "[ERROR] Successful Phase B smoke test is required before submission" >&2
  exit 2
}
[[ -s "$manifest" ]] || {
  echo "[ERROR] Missing Phase B preflight manifest: $manifest" >&2
  exit 2
}

# Each completed run retains four ~336 MB candidate checkpoints. Keep 2 GB free
# for logs, repository work, and filesystem headroom; submit only a safe batch.
checkpoint_bytes_per_run=1344472356
reserve_bytes=$((2 * 1024 * 1024 * 1024))
available_bytes="$(df -PB1 . | awk 'NR==2 {print $4}')"
capacity_runs=0
if (( available_bytes > reserve_bytes )); then
  capacity_runs=$(( (available_bytes - reserve_bytes) / checkpoint_bytes_per_run ))
fi
(( capacity_runs > 9 )) && capacity_runs=9
pending_runs=0
if [[ -s "$registry" ]]; then
  while IFS=$'\t' read -r demos model; do
    [[ -n "$demos" && -n "$model" ]] || continue
    if [[ ! -e "results/act/phase_b/${demos}demos/${model}/train_seed0/.training_complete" ]]; then
      pending_runs=$((pending_runs + 1))
    fi
  done < <(awk -F '\t' 'NR>1 && !seen[$3]++ {print $4 "\t" $5}' "$registry")
fi
safe_jobs=$((capacity_runs - pending_runs))
(( safe_jobs < 0 )) && safe_jobs=0
requested_jobs="${MAX_JOBS:-$safe_jobs}"
[[ "$requested_jobs" =~ ^[0-9]+$ ]] || {
  echo "[ERROR] MAX_JOBS must be a non-negative integer" >&2
  exit 2
}
if (( requested_jobs > safe_jobs )); then
  echo "[ERROR] MAX_JOBS=$requested_jobs exceeds storage-safe limit $safe_jobs" >&2
  exit 3
fi
if (( requested_jobs == 0 )); then
  echo "[ERROR] No run can be submitted while retaining the 2 GB safety reserve" >&2
  exit 3
fi

mkdir -p "$(dirname "$registry")"
if [[ ! -e "$registry" ]]; then
  printf 'submitted_at\tjob_id\trun_name\tdemo_count\tmodel\tcamera_names\n' > "$registry"
fi

submitted=0
for demos in 5 10 25; do
  for spec in 'single_top:top' 'single_side:side' 'multi:top,side'; do
    model="${spec%%:*}"
    cameras="${spec#*:}"
    run_name="phase_b_${demos}demos_${model}_seed0"
    result_dir="results/act/phase_b/${demos}demos/${model}/train_seed0"
    if [[ -e "$result_dir/.training_complete" ]]; then
      echo "[SKIP] already complete: $run_name"
      continue
    fi
    if awk -F '\t' -v run="$run_name" 'NR>1 && $3==run {found=1} END {exit !found}' "$registry"; then
      echo "[SKIP] already registered as submitted: $run_name"
      continue
    fi
    (( submitted >= requested_jobs )) && break 2
    # Pass the view as a comma-free symbolic name; the PBS script resolves and
    # verifies the exact Clean-50 camera order.
    job_id="$(qsub -v VIEW_CONFIG="$model",DEMO_COUNT="$demos",TRAIN_SEED=0 \
      scripts/run_phase_b_train.pbs)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "$job_id" "$run_name" "$demos" "$model" "$cameras" \
      >> "$registry"
    echo "[SUBMITTED] $run_name -> $job_id"
    submitted=$((submitted + 1))
  done
done

echo "[INFO] submitted=$submitted additional_storage_safe_limit=$safe_jobs pending_runs=$pending_runs available_bytes=$available_bytes"
echo "[INFO] registry=$registry"
