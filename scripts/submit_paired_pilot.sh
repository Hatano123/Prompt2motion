#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -s archives/datasets/act_sim_insertion_scripted_top_side_seed0_10episodes.zip ]]; then
  data_job="$(qsub scripts/run_prepare_paired_dataset.pbs)"
  dependency="afterok:$data_job"
  echo "paired_dataset=$data_job"
else
  dependency=""
  echo "paired_dataset=existing"
fi

for spec in 'single_top:top' 'single_side:side' 'multi:top,side'; do
  model_name="${spec%%:*}"
  camera_names="${spec#*:}"
  export MODEL_NAME="$model_name" CAMERA_NAMES="$camera_names"
  if [[ -n "$dependency" ]]; then
    train_job="$(qsub -W depend="$dependency" -v MODEL_NAME,CAMERA_NAMES scripts/run_paired_pilot_train.pbs)"
  else
    train_job="$(qsub -v MODEL_NAME,CAMERA_NAMES scripts/run_paired_pilot_train.pbs)"
  fi
  eval_job="$(qsub -W depend="afterok:$train_job" -v MODEL_NAME,CAMERA_NAMES scripts/run_paired_pilot_eval.pbs)"
  echo "$model_name train=$train_job eval=$eval_job"
done
