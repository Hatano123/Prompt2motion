#!/bin/bash

normalize_pbs_gpu() {
    local pbs_visible="${CUDA_VISIBLE_DEVICES-}"
    case "$pbs_visible" in
        [0-9]*-*)
            export CUDA_VISIBLE_DEVICES="${pbs_visible%%-*}"
            ;;
        ''|*[!0-9,]*)
            echo "[ERROR] unsupported CUDA_VISIBLE_DEVICES=${pbs_visible:-<unset>}" >&2
            return 4
            ;;
    esac
    echo "[INFO] PBS CUDA_VISIBLE_DEVICES=${pbs_visible:-<unset>}"
    echo "[INFO] normalized CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

prepare_act_runtime() {
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

    "$act_python" -m pip install --no-cache-dir \
        torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/rocm6.4
    "$act_python" -m pip install --no-cache-dir -r "$submit_dir/requirements-act.txt"
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
