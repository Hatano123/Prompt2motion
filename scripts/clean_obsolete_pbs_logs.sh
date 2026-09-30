#!/usr/bin/env bash
# Remove only root-level PBS stdout/stderr files for jobs that are no longer active.
# Experimental artifacts under results/ are intentionally never considered here.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
apply=false

if [[ ${1:-} == "--apply" && $# -eq 1 ]]; then
  apply=true
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--apply]" >&2
  exit 2
fi

declare -A active_job_ids=()
while IFS= read -r job_id; do
  [[ -n "$job_id" ]] || continue
  active_job_ids["${job_id%%.*}"]=1
done < <(qstat -u "${USER:?USER must be set}" 2>/dev/null | awk 'NR > 2 {print $1}')

targets=()
while IFS= read -r -d '' path; do
  name=${path##*/}
  if [[ $name =~ \.[oe]([0-9]+)$ ]]; then
    job_id=${BASH_REMATCH[1]}
    [[ -v "active_job_ids[$job_id]" ]] || targets+=("$path")
  fi
done < <(find "$repo_root" -maxdepth 1 -type f \( -name '*.o[0-9]*' -o -name '*.e[0-9]*' \) -print0)

bytes=0
for path in "${targets[@]}"; do
  size=$(stat -c '%s' "$path")
  bytes=$((bytes + size))
  printf '%s\t%s\n' "$size" "$path"
done
printf 'eligible_logs=%d bytes=%d\n' "${#targets[@]}" "$bytes"

if ! "$apply"; then
  echo 'dry-run: no files removed (rerun with --apply after reviewing this list).'
  exit 0
fi

for path in "${targets[@]}"; do
  rm -- "$path"
done
printf 'removed_logs=%d bytes=%d\n' "${#targets[@]}" "$bytes"
