#!/bin/bash

normalize_pbs_gpu() {
    local pbs_visible="${CUDA_VISIBLE_DEVICES-}"
    local source_name="CUDA_VISIBLE_DEVICES"
    local source_value="$pbs_visible"

    # Some PBS hook versions expose the allocated AMD GPU through a ROCm
    # variable but omit CUDA_VISIBLE_DEVICES.  Reuse only an unambiguous,
    # single numeric allocation; never guess a physical GPU index.
    if [[ -z "$source_value" ]]; then
        local candidate_name candidate_value
        for candidate_name in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
            candidate_value="${!candidate_name-}"
            if [[ "$candidate_value" =~ ^[0-9]+(-.*)?$ ]]; then
                source_name="$candidate_name"
                source_value="$candidate_value"
                break
            fi
        done
    fi

    # PBS installations may instead provide one allocated device in a GPU
    # file.  Accept it only when there is exactly one non-empty line and that
    # line has an explicit trailing GPU index (for example, host-gpu0).
    if [[ -z "$source_value" && -n "${PBS_GPUFILE-}" && -r "$PBS_GPUFILE" ]]; then
        local gpu_file_lines gpu_file_index
        gpu_file_lines="$(awk 'NF {count++} END {print count+0}' "$PBS_GPUFILE")"
        if [[ "$gpu_file_lines" == 1 ]]; then
            gpu_file_index="$(sed -nE 's/.*([Gg][Pp][Uu]|device)[^0-9]*([0-9]+)$/\2/p' "$PBS_GPUFILE")"
            if [[ "$gpu_file_index" =~ ^[0-9]+$ ]]; then
                source_name="PBS_GPUFILE"
                source_value="$gpu_file_index"
            fi
        fi
    fi

    if [[ -z "$source_value" ]]; then
        echo "[WARN] PBS did not expose a GPU visibility variable; deferring to the runtime single-device check"
        echo "[INFO] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES-<unset>}"
        echo "[INFO] ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES-<unset>}"
        echo "[INFO] GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL-<unset>}"
        echo "[INFO] PBS_GPUFILE=${PBS_GPUFILE-<unset>}"
        return 0
    fi

    pbs_visible="$source_value"
    case "$pbs_visible" in
      [0-9]*-*)
            pbs_visible="${pbs_visible%%-*}"
            ;;
        *[!0-9]*|'')
            echo "[ERROR] unsupported single-GPU allocation from $source_name=${source_value:-<unset>}" >&2
            return 4
            ;;
    esac
    export CUDA_VISIBLE_DEVICES="$pbs_visible"
    echo "[INFO] PBS GPU allocation source: $source_name=$source_value"
    echo "[INFO] normalized CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

prepare_act_runtime() {
    local runtime_mode="${1:-gpu}"
    if [[ "$runtime_mode" != "gpu" && "$runtime_mode" != "cpu" ]]; then
        echo "[ERROR] runtime mode must be gpu or cpu: $runtime_mode" >&2
        return 4
    fi
    repo_stage_dir="$workdir/repo"
    runtime_tmp="$workdir/tmp"
    mkdir -p "$repo_stage_dir" "$runtime_tmp"

    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete --exclude '.git/' "$submit_dir/repos/act/" "$repo_stage_dir/"
    else
        cp -a "$submit_dir/repos/act/." "$repo_stage_dir/"
        rm -rf -- "$repo_stage_dir/.git"
    fi

    export MAMBA_ROOT_PREFIX="$workdir/mamba-root"
    export PIP_CACHE_DIR="$runtime_tmp/pip-cache"
    export XDG_CACHE_HOME="$runtime_tmp/xdg-cache"
    export MPLCONFIGDIR="$runtime_tmp/matplotlib"
    export TMPDIR="$runtime_tmp"
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

    "$submit_dir/tools/micromamba" create -y -n act -f "$submit_dir/environment-act-rocm.yml"
    act_python="$MAMBA_ROOT_PREFIX/envs/act/bin/python"
    export LD_LIBRARY_PATH="$MAMBA_ROOT_PREFIX/envs/act/lib:${LD_LIBRARY_PATH:-}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$MAMBA_ROOT_PREFIX/envs/act/share/glvnd/egl_vendor.d/50_mesa.json"
    export LIBGL_ALWAYS_SOFTWARE=1

    "$act_python" -m pip install --no-cache-dir -r "$submit_dir/requirements-act.txt"
    if [[ "$runtime_mode" == "cpu" ]]; then
        "$act_python" -m pip install --no-cache-dir torch==2.8.0 \
            --index-url https://download.pytorch.org/whl/cpu
        echo "[INFO] prepared CPU-only ACT data runtime"
        return 0
    fi

    "$act_python" -m pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/rocm6.4
    "$act_python" -m pip install --no-cache-dir --no-deps -e "$repo_stage_dir/detr"
    "$act_python" - <<'PY_RUNTIME'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError('exactly one PBS-allocated GPU must be visible')
print('torch:', torch.__version__)
print('HIP:', torch.version.hip)
print('GPU:', torch.cuda.get_device_name(0))
print('GPU checksum:', (torch.ones((512, 512), device='cuda') @ torch.ones((512, 512), device='cuda')).sum().item())
PY_RUNTIME
}

stage_act50_dataset() {
    dataset_root="$workdir/dataset"
    mkdir -p "$dataset_root"
    unzip -q "$submit_dir/archives/datasets/act_sim_transfer_cube_scripted_50episodes.zip" -d "$dataset_root"
    training_dataset="$dataset_root/sim_transfer_cube_scripted"
    for episode_idx in $(seq 0 49); do
        if [[ ! -s "$training_dataset/episode_${episode_idx}.hdf5" ]]; then
            echo "[ERROR] missing episode_${episode_idx}.hdf5" >&2
            return 5
        fi
    done
    episode_count="$(find "$training_dataset" -maxdepth 1 -type f -name 'episode_*.hdf5' | wc -l)"
    if [[ "$episode_count" != "50" ]]; then
        echo "[ERROR] expected 50 episodes, found $episode_count" >&2
        return 5
    fi
    export ACT_DATA_DIR="$dataset_root"
    export ACT_NUM_EPISODES=50
    echo "[INFO] staged 50-demo dataset: $training_dataset"
}
