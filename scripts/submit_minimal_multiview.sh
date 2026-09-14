#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
job_id="$(qsub -v TRAIN_EPISODES="${TRAIN_EPISODES:-2}",TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}",EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-1}",SAVE_EVERY="${SAVE_EVERY:-1}" scripts/run_minimal_multiview.pbs)"
echo "minimal_multiview=$job_id"
