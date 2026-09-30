#!/bin/bash
# Shared, side-effect-free path and checkpoint resolution for Phases B--D.

experiment_source_dir() {
  local root=$1 stage=$2 method=$3 demos=$4 model=$5 seed=$6
  case "$stage" in
    phase_b)
      local canonical="$root/results/act/phase_b/${demos}demos/$model/train_seed${seed}"
      local rebuild="$root/results/act/phase_b_rebuild/${demos}demos/$model/train_seed${seed}"
      if [[ -e "$rebuild/.training_complete" ]]; then printf '%s\n' "$rebuild"
      else printf '%s\n' "$canonical"; fi ;;
    phase_d)
      printf '%s/results/act/phase_d/%s/%sdemos/%s/train_seed%s\n' \
        "$root" "$method" "$demos" "$model" "$seed" ;;
    *) echo "[ERROR] unsupported source stage: $stage" >&2; return 4 ;;
  esac
}

experiment_selection_json() {
  local source_dir=$1
  if [[ -s "$source_dir/checkpoint_selection_deterministic/validation_summary.json" ]]; then
    printf '%s/checkpoint_selection_deterministic/validation_summary.json\n' "$source_dir"
  elif [[ -s "$source_dir/validation_summary.json" ]]; then
    printf '%s/validation_summary.json\n' "$source_dir"
  else
    echo "[ERROR] no checkpoint-selection summary under $source_dir" >&2
    return 5
  fi
}

require_free_space() {
  local path=$1 required_bytes=$2 reserve_bytes=$3
  local available
  available=$(df -PB1 "$path" | awk 'NR==2 {print $4}')
  if (( available < required_bytes + reserve_bytes )); then
    echo "[ERROR] insufficient storage: available=$available required=$required_bytes reserve=$reserve_bytes" >&2
    return 6
  fi
}
