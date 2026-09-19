#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-paired_pilot}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
test -s archives/datasets/act_sim_insertion_scripted_top_side_seed0_10episodes.zip
test -s archives/environments/sim_insertion_eval_seed1000_50.json

for spec in 'single_top:top' 'single_side:side' 'multi:top,side'; do
  export MODEL_NAME="${spec%%:*}" CAMERA_NAMES="${spec#*:}"
  export TRAIN_SEED=0
  seed0_dir="results/act/$EXPERIMENT_NAME/$MODEL_NAME/train_seed0"
  if [[ -s "$seed0_dir/selected_policy.ckpt" && ! -s "$seed0_dir/eval_robustness/.complete" ]]; then
    job_id="$(qsub -v MODEL_NAME,CAMERA_NAMES,TRAIN_SEED,EXPERIMENT_NAME,BATCH_SIZE scripts/run_paired_robustness_eval.pbs)"
    echo "$MODEL_NAME seed=0 robustness=$job_id"
  fi
  for TRAIN_SEED in 1 2; do
    export TRAIN_SEED
    result_dir="results/act/$EXPERIMENT_NAME/$MODEL_NAME/train_seed${TRAIN_SEED}"
    if [[ -s "$result_dir/selected_policy.ckpt" ]]; then
      eval_job="$(qsub -v MODEL_NAME,CAMERA_NAMES,TRAIN_SEED,EXPERIMENT_NAME,BATCH_SIZE scripts/run_paired_robustness_eval.pbs)"
      echo "$MODEL_NAME seed=$TRAIN_SEED training=existing robustness=$eval_job"
    else
      train_job="$(qsub -v MODEL_NAME,CAMERA_NAMES,TRAIN_SEED,EXPERIMENT_NAME,BATCH_SIZE scripts/run_paired_pilot_train.pbs)"
      eval_job="$(qsub -W depend=afterok:$train_job -v MODEL_NAME,CAMERA_NAMES,TRAIN_SEED,EXPERIMENT_NAME,BATCH_SIZE scripts/run_paired_robustness_eval.pbs)"
      echo "$MODEL_NAME seed=$TRAIN_SEED training=$train_job robustness=$eval_job"
    fi
  done
done
