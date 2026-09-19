#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export BATCH_SIZE="${BATCH_SIZE:-8}"
export DELETE_POLICY_AFTER_TEST="${DELETE_POLICY_AFTER_TEST:-1}"
archive="archives/datasets/act_sim_insertion_scripted_top_side_seed0_50episodes.zip"
validation_manifest="archives/environments/sim_insertion_validation_seed1100_50.json"
test_manifest="archives/environments/sim_insertion_test_seed1200_100.json"

if [[ ! -s "$archive" || ! -s "$validation_manifest" || ! -s "$test_manifest" ]]; then
  data_job="$(qsub scripts/run_prepare_clean50_dataset.pbs)"
  chain_dependency="afterok:$data_job"
  echo "clean50_dataset=$data_job"
else
  chain_dependency=""
  echo "clean50_dataset=existing"
fi

for spec in 'single_top:top' 'single_side:side' 'multi:top,side'; do
  export MODEL_NAME="${spec%%:*}" CAMERA_NAMES="${spec#*:}"
  for TRAIN_SEED in 0 1 2; do
    export TRAIN_SEED
    result_dir="results/act/clean50/$MODEL_NAME/train_seed$TRAIN_SEED"
    if [[ -e "$result_dir/eval_test/.complete" ]]; then
      echo "$MODEL_NAME seed=$TRAIN_SEED test=existing"
      continue
    fi
    if [[ -e "$result_dir/.training_complete" && -s "$result_dir/selected_policy.ckpt" ]]; then
      train_job=""
      echo "$MODEL_NAME seed=$TRAIN_SEED training=existing"
    elif [[ -n "$chain_dependency" ]]; then
      train_job="$(qsub -W depend="$chain_dependency" -v MODEL_NAME,CAMERA_NAMES,TRAIN_SEED,BATCH_SIZE scripts/run_clean50_train.pbs)"
      echo "$MODEL_NAME seed=$TRAIN_SEED training=$train_job"
    else
      train_job="$(qsub -v MODEL_NAME,CAMERA_NAMES,TRAIN_SEED,BATCH_SIZE scripts/run_clean50_train.pbs)"
      echo "$MODEL_NAME seed=$TRAIN_SEED training=$train_job"
    fi

    if [[ -n "$train_job" ]]; then
      test_job="$(qsub -W depend="afterok:$train_job" -v MODEL_NAME,CAMERA_NAMES,TRAIN_SEED,BATCH_SIZE,DELETE_POLICY_AFTER_TEST scripts/run_clean50_test.pbs)"
      echo "$MODEL_NAME seed=$TRAIN_SEED test=$test_job"
    elif [[ -n "$chain_dependency" ]]; then
      test_job="$(qsub -W depend="$chain_dependency" -v MODEL_NAME,CAMERA_NAMES,TRAIN_SEED,BATCH_SIZE,DELETE_POLICY_AFTER_TEST scripts/run_clean50_test.pbs)"
      echo "$MODEL_NAME seed=$TRAIN_SEED test=$test_job"
    else
      test_job="$(qsub -v MODEL_NAME,CAMERA_NAMES,TRAIN_SEED,BATCH_SIZE,DELETE_POLICY_AFTER_TEST scripts/run_clean50_test.pbs)"
      echo "$MODEL_NAME seed=$TRAIN_SEED test=$test_job"
    fi
    chain_dependency="afterok:$test_job"
  done
done
