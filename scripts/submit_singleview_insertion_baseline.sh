#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
job_id="$(qsub scripts/run_singleview_insertion_baseline.pbs)"
echo "singleview_insertion_baseline=$job_id"
