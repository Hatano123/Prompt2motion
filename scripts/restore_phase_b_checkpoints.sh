#!/bin/bash
# Restore one Phase-B seed-0 checkpoint set from a split tar.zst backup.
# Existing files are never overwritten or removed.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/restore_phase_b_checkpoints.sh \
    --archive-dir DIR --demo-count {5|10} \
    --view-config {single_top|single_side|multi} [options]

Options:
  --repo-root DIR       Repository root (default: parent of this script)
  --output-namespace N  phase_b or phase_b_rebuild (default: phase_b)
  --reserve-bytes N     Free-space reserve on the destination filesystem
                        (default: 2147483648, i.e. 2 GiB)
  --dry-run             Verify inputs and report the planned restore only
  -h, --help            Show this help

The archive directory must contain SHA256SUMS.txt and exactly one archive
whose first part is named *.tar.zst.part-000. All listed archive parts are
verified before extraction. The destination checkpoint_manifest.json is the
authority for member names, sizes, and SHA-256 digests.
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

archive_dir=""
demo_count=""
view_config=""
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_namespace="phase_b"
reserve_bytes=2147483648
dry_run=0

while (($#)); do
  case "$1" in
    --archive-dir) [[ $# -ge 2 ]] || die "$1 requires a value"; archive_dir=$2; shift 2 ;;
    --demo-count) [[ $# -ge 2 ]] || die "$1 requires a value"; demo_count=$2; shift 2 ;;
    --view-config) [[ $# -ge 2 ]] || die "$1 requires a value"; view_config=$2; shift 2 ;;
    --repo-root) [[ $# -ge 2 ]] || die "$1 requires a value"; repo_root=$2; shift 2 ;;
    --output-namespace) [[ $# -ge 2 ]] || die "$1 requires a value"; output_namespace=$2; shift 2 ;;
    --reserve-bytes) [[ $# -ge 2 ]] || die "$1 requires a value"; reserve_bytes=$2; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$archive_dir" ]] || die "--archive-dir is required"
case "$demo_count" in 5|10) ;; *) die "--demo-count must be 5 or 10" ;; esac
case "$view_config" in single_top|single_side|multi) ;; *) die "invalid --view-config: $view_config" ;; esac
case "$output_namespace" in phase_b|phase_b_rebuild) ;; *) die "invalid --output-namespace: $output_namespace" ;; esac
[[ "$reserve_bytes" =~ ^[0-9]+$ ]] || die "--reserve-bytes must be a non-negative integer"

for command_name in sha256sum zstd tar jq stat df mktemp; do
  command -v "$command_name" >/dev/null || die "required command is unavailable: $command_name"
done

archive_dir="$(cd "$archive_dir" && pwd)"
repo_root="$(cd "$repo_root" && pwd)"
checksum_file="$archive_dir/SHA256SUMS.txt"
[[ -s "$checksum_file" ]] || die "missing checksum file: $checksum_file"

mapfile -t part_zero_candidates < <(find "$archive_dir" -maxdepth 1 -type f -name '*.tar.zst.part-000' -print | sort)
[[ ${#part_zero_candidates[@]} -eq 1 ]] || \
  die "expected exactly one *.tar.zst.part-000 in $archive_dir"
archive_prefix="${part_zero_candidates[0]%-000}"
mapfile -t archive_parts < <(find "$archive_dir" -maxdepth 1 -type f -name "$(basename "$archive_prefix")-[0-9][0-9][0-9]" -print | sort)
[[ ${#archive_parts[@]} -gt 0 ]] || die "no archive parts found for $archive_prefix"

for ((index=0; index<${#archive_parts[@]}; index++)); do
  expected=$(printf '%s-%03d' "$archive_prefix" "$index")
  [[ "${archive_parts[$index]}" == "$expected" ]] || \
    die "archive parts are not contiguous: expected $expected"
done

echo "[INFO] verifying ${#archive_parts[@]} archive parts"
(
  cd "$archive_dir"
  sha256sum --check --strict "$(basename "$checksum_file")"
) >/dev/null

archive_run="results/act/phase_b/${demo_count}demos/${view_config}/train_seed0"
relative_run="results/act/${output_namespace}/${demo_count}demos/${view_config}/train_seed0"
source_run="$repo_root/$archive_run"
run_dir="$repo_root/$relative_run"
manifest="$source_run/checkpoint_manifest.json"
[[ -s "$manifest" ]] || die "missing destination manifest: $manifest"

declare -a member_paths=()
declare -a expected_sizes=()
declare -a expected_hashes=()
declare -a restore_indexes=()
while IFS=$'\t' read -r name size digest; do
  [[ "$name" =~ ^policy_epoch_(500|1000|1500|2000)_seed_0\.ckpt$ ]] || \
    die "unsafe checkpoint name in manifest: $name"
  [[ "$size" =~ ^[0-9]+$ ]] || die "invalid byte count for $name"
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || die "invalid SHA-256 for $name"
  member_paths+=("$archive_run/$name")
  expected_sizes+=("$size")
  expected_hashes+=("$digest")
done < <(jq -r '.checkpoints[] | [.name, (.bytes | tostring), .sha256] | @tsv' "$manifest")
[[ ${#member_paths[@]} -eq 4 ]] || die "manifest must describe exactly four checkpoints"

missing_bytes=0
for index in "${!member_paths[@]}"; do
  checkpoint_name="${member_paths[$index]##*/}"
  destination="$run_dir/$checkpoint_name"
  if [[ -e "$destination" ]]; then
    [[ -f "$destination" ]] || die "destination exists and is not a regular file: $destination"
    actual_size=$(stat -c %s "$destination")
    actual_hash=$(sha256sum "$destination" | awk '{print $1}')
    [[ "$actual_size" == "${expected_sizes[$index]}" && "$actual_hash" == "${expected_hashes[$index]}" ]] || \
      die "existing checkpoint does not match the manifest; refusing to overwrite: $destination"
    echo "[SKIP] already verified: $relative_run/$checkpoint_name"
  else
    restore_indexes+=("$index")
    missing_bytes=$((missing_bytes + expected_sizes[index]))
  fi
done

if [[ ${#restore_indexes[@]} -eq 0 ]]; then
  echo "[OK] all four checkpoints are already restored and verified"
  exit 0
fi

capacity_parent="$run_dir"
while [[ ! -e "$capacity_parent" ]]; do capacity_parent=$(dirname "$capacity_parent"); done
available_bytes=$(df -PB1 "$capacity_parent" | awk 'NR==2 {print $4}')
required_bytes=$((missing_bytes + reserve_bytes))
(( available_bytes >= required_bytes )) || \
  die "insufficient destination space: available=$available_bytes restore=$missing_bytes reserve=$reserve_bytes"

echo "[PLAN] run=$relative_run files=${#restore_indexes[@]} bytes=$missing_bytes reserve=$reserve_bytes"
if ((dry_run)); then
  echo "[OK] dry-run completed; no checkpoints were written"
  exit 0
fi

mkdir -p "$run_dir"
stage_dir=""
declare -a destination_temps=()
cleanup() {
  status=$?
  trap - EXIT
  for temporary in "${destination_temps[@]:-}"; do
    [[ -n "$temporary" ]] && rm -f -- "$temporary"
  done
  [[ -z "$stage_dir" ]] || rm -rf -- "$stage_dir"
  exit "$status"
}
trap cleanup EXIT INT TERM

# A rebuild namespace receives the immutable top-level training metadata from
# the original run. Selection/test artifacts and directories are intentionally
# excluded so an earlier failed evaluation cannot contaminate the rebuild.
if [[ "$output_namespace" == phase_b_rebuild ]]; then
  while IFS= read -r -d '' source_metadata; do
    metadata_name="${source_metadata##*/}"
    [[ "$metadata_name" == policy_epoch_*_seed_0.ckpt ]] && continue
    [[ "$metadata_name" == .training_complete ]] && continue
    destination_metadata="$run_dir/$metadata_name"
    if [[ -e "$destination_metadata" ]]; then
      [[ -f "$destination_metadata" ]] || die "metadata destination is not a regular file: $destination_metadata"
      cmp -s "$source_metadata" "$destination_metadata" || \
        die "rebuild metadata differs; refusing to overwrite: $destination_metadata"
      continue
    fi
    metadata_temporary="${destination_metadata}.restore.$$"
    destination_temps+=("$metadata_temporary")
    cp --preserve=timestamps -- "$source_metadata" "$metadata_temporary"
    cmp -s "$source_metadata" "$metadata_temporary" || die "metadata copy verification failed: $metadata_temporary"
    mv --no-clobber -- "$metadata_temporary" "$destination_metadata"
  done < <(find "$source_run" -maxdepth 1 -type f -print0)
fi

stage_dir=$(mktemp -d "${TMPDIR:-/tmp}/phase_b_restore_XXXXXX")

declare -a members_to_extract=()
for index in "${restore_indexes[@]}"; do
  members_to_extract+=("${member_paths[$index]}")
done

echo "[INFO] extracting ${#members_to_extract[@]} checkpoints into temporary storage"
cat "${archive_parts[@]}" | zstd --decompress --stdout | \
  tar -xf - -C "$stage_dir" -- "${members_to_extract[@]}"

for index in "${restore_indexes[@]}"; do
  staged="$stage_dir/${member_paths[$index]}"
  [[ -f "$staged" ]] || die "archive member was not extracted: ${member_paths[$index]}"
  actual_size=$(stat -c %s "$staged")
  actual_hash=$(sha256sum "$staged" | awk '{print $1}')
  [[ "$actual_size" == "${expected_sizes[$index]}" && "$actual_hash" == "${expected_hashes[$index]}" ]] || \
    die "restored checkpoint failed manifest verification: ${member_paths[$index]}"
done

for index in "${restore_indexes[@]}"; do
  checkpoint_name="${member_paths[$index]##*/}"
  destination="$run_dir/$checkpoint_name"
  staged="$stage_dir/${member_paths[$index]}"
  temporary="${destination}.restore.$$"
  destination_temps+=("$temporary")
  [[ ! -e "$destination" ]] || die "destination appeared during restore; refusing to overwrite: $destination"
  cp --preserve=timestamps -- "$staged" "$temporary"
  copied_hash=$(sha256sum "$temporary" | awk '{print $1}')
  [[ "$copied_hash" == "${expected_hashes[$index]}" ]] || die "destination copy verification failed: $temporary"
  mv --no-clobber -- "$temporary" "$destination"
  [[ -f "$destination" && ! -e "$temporary" ]] || die "atomic destination install failed: $destination"
  echo "[RESTORED] $relative_run/$checkpoint_name"
done

if [[ "$output_namespace" == phase_b_rebuild ]]; then
  source_marker="$source_run/.training_complete"
  destination_marker="$run_dir/.training_complete"
  [[ -f "$source_marker" ]] || die "source training marker is missing: $source_marker"
  if [[ ! -e "$destination_marker" ]]; then
    marker_temporary="${destination_marker}.restore.$$"
    destination_temps+=("$marker_temporary")
    cp --preserve=timestamps -- "$source_marker" "$marker_temporary"
    mv --no-clobber -- "$marker_temporary" "$destination_marker"
  fi
fi

echo "[OK] restored and verified Phase-B checkpoints: demos=$demo_count view=$view_config seed=0"
