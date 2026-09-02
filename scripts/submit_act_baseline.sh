#!/bin/bash
set -euo pipefail

submit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$submit_dir"
experiment_name="${EXPERIMENT_NAME:-act50_baseline}"
total_epochs="${TOTAL_EPOCHS:-2000}"
epochs_per_job="${EPOCHS_PER_JOB:-500}"
eval_rollouts="${EVAL_ROLLOUTS:-50}"

if (( total_epochs % epochs_per_job != 0 )); then
    echo "TOTAL_EPOCHS must be divisible by EPOCHS_PER_JOB" >&2
    exit 2
fi

previous_job=""
for completed_epoch in $(seq "$epochs_per_job" "$epochs_per_job" "$total_epochs"); do
    dependency=()
    if [[ -n "$previous_job" ]]; then
        dependency=(-W "depend=afterok:$previous_job")
    fi
    train_job="$(qsub "${dependency[@]}" \
        -v "EXPERIMENT_NAME=$experiment_name,TOTAL_EPOCHS=$total_epochs,EPOCHS_PER_JOB=$epochs_per_job" \
        scripts/run_act_baseline.pbs)"
    eval_job="$(qsub -W "depend=afterok:$train_job" \
        -v "EXPERIMENT_NAME=$experiment_name,EVAL_EPOCH=$completed_epoch,EVAL_ROLLOUTS=$eval_rollouts" \
        scripts/run_act_eval.pbs)"
    printf 'epoch=%s train_job=%s eval_job=%s\n' "$completed_epoch" "$train_job" "$eval_job"
    previous_job="$eval_job"
done
