#!/bin/bash
# Create a reproducible tar.zst archive from an explicit repository-relative list.
# This script never removes or changes a listed source path.

set -euo pipefail

usage() {
  echo "Usage: $0 --manifest FILE --output FILE.tar.zst [--repo-root DIR]" >&2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest=""
output=""
while (($#)); do
  case "$1" in
    --manifest) manifest=${2:?}; shift 2 ;;
    --output) output=${2:?}; shift 2 ;;
    --repo-root) repo_root=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$manifest" && -n "$output" ]] || { usage; exit 2; }
repo_root="$(cd "$repo_root" && pwd)"
manifest="$(realpath "$manifest")"
[[ -f "$manifest" ]] || { echo "[ERROR] manifest not found: $manifest" >&2; exit 2; }
[[ ! -e "$output" && ! -e "$output.sha256" && ! -e "$output.contents.tsv" ]] || {
  echo "[ERROR] refusing to overwrite archive or sidecar: $output" >&2
  exit 3
}

declare -a paths=()
while IFS= read -r path; do
  [[ -z "$path" || "$path" == \#* ]] && continue
  case "$path" in
    /*|../*|*/../*|*/..) echo "[ERROR] unsafe path: $path" >&2; exit 4 ;;
  esac
  [[ -e "$repo_root/$path" ]] || { echo "[ERROR] missing source: $path" >&2; exit 4; }
  paths+=("$path")
done < "$manifest"
[[ ${#paths[@]} -gt 0 ]] || { echo "[ERROR] manifest contains no paths" >&2; exit 4; }

mkdir -p "$(dirname "$output")"
temporary="${output}.partial.$$"
trap 'rm -f -- "$temporary"' EXIT INT TERM

(
  cd "$repo_root"
  tar --sort=name --mtime='UTC 2026-09-29' --owner=0 --group=0 --numeric-owner \
    -cf - -- "${paths[@]}"
) | zstd -T0 -10 -o "$temporary"
mv --no-clobber -- "$temporary" "$output"
trap - EXIT INT TERM

sha256sum "$output" > "$output.sha256"
(
  cd "$repo_root"
  for path in "${paths[@]}"; do
    bytes=$(du -sb -- "$path" | awk '{print $1}')
    printf '%s\t%s\n' "$bytes" "$path"
  done
) > "$output.contents.tsv"

echo "[OK] archive=$output"
echo "[OK] sha256=$(awk '{print $1}' "$output.sha256")"
