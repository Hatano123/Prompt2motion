#!/bin/bash
set -euo pipefail

submit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$submit_dir"
experiment_name="${EXPERIMENT_NAME:-act50_resume_smoke}"

first_job="$(qsub \
    -v "EXPERIMENT_NAME=$experiment_name,TOTAL_EPOCHS=2,EPOCHS_PER_JOB=1,SAVE_EVERY=1,BATCH_SIZE=8" \
    scripts/run_act_baseline.pbs)"
second_job="$(qsub -W "depend=afterok:$first_job" \
    -v "EXPERIMENT_NAME=$experiment_name,TOTAL_EPOCHS=2,EPOCHS_PER_JOB=1,SAVE_EVERY=1,BATCH_SIZE=8" \
    scripts/run_act_baseline.pbs)"
eval_job="$(qsub -W "depend=afterok:$second_job" \
    -v "EXPERIMENT_NAME=$experiment_name,EVAL_EPOCH=2,EVAL_ROLLOUTS=1,BATCH_SIZE=8" \
    scripts/run_act_eval.pbs)"
printf 'smoke_train_1=%s\nsmoke_resume_2=%s\nsmoke_eval=%s\n' "$first_job" "$second_job" "$eval_job"
