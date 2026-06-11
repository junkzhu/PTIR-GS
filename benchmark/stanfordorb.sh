#!/usr/bin/env bash

set -euo pipefail

CUDA_DEVICES="0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
DATA_ROOT="$REPO_ROOT/data/StanfordOrb"
OUT_DIR="outputs/stanfordorb"
CONFIG_NAME="apps/nerf_synthetic_3dgrt.yaml"
INVERSION_CONFIG_NAME="inversions/nerf_synthetic_3dgptir.yaml"
INVERSION_OUT_DIR=""
RUN_INVERSION=true
DATASET_CONFIG="stanfordorb"
FORCE_TRAIN=false
INITIALIZATION_NUM_GAUSSIANS="50_000"
INITIALIZATION_XYZ_MAX="0.5"
INITIALIZATION_XYZ_MIN="-0.5"
SCENES=(
    cactus_scene001 gnome_scene003 car_scene002
)
EXTRA_ARGS=()
INVERSION_EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 --cuda_device 0,1,2,3 [options] [-- extra hydra args]

Options:
  --cuda_device DEVICES      Comma-separated GPU ids, e.g. 0,1,2,3.
  --data_root PATH           StanfordORB dataset root. Default: $DATA_ROOT
  --out_dir PATH             Output directory. Default: $OUT_DIR
  --config_name NAME         Hydra config name. Default: $CONFIG_NAME
  --inversion_config_name NAME
                             Hydra config for PTIR inversion. Default: $INVERSION_CONFIG_NAME
  --inversion_out_dir PATH   PTIR inversion output directory. Default: same as --out_dir.
  --inversion_args "ARGS"    Extra Hydra args only for PTIR inversion.
  --no_inversion             Only run/skip stage1 training; do not run PTIR inversion.
  --dataset_config NAME      Hydra dataset config. Default: $DATASET_CONFIG
  --scenes "A B C"           Space-separated scene list. Default: all StanfordORB scenes found here.
  --force_train              Run training even if ckpt_last.pt already exists.
  -h, --help                 Show this help.

Examples:
  $0 --cuda_device 0,1,2,3
  $0 --cuda_device 0 --scenes "baking_scene001" -- n_iterations=7000
  $0 --cuda_device 0 --scenes "baking_scene001" --no_inversion
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda_device)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --config_name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --inversion_config_name)
            INVERSION_CONFIG_NAME="$2"
            shift 2
            ;;
        --inversion_out_dir)
            INVERSION_OUT_DIR="$2"
            shift 2
            ;;
        --inversion_args)
            read -r -a INVERSION_EXTRA_ARGS <<< "$2"
            shift 2
            ;;
        --no_inversion)
            RUN_INVERSION=false
            shift
            ;;
        --dataset_config)
            DATASET_CONFIG="$2"
            shift 2
            ;;
        --scenes)
            read -r -a SCENES <<< "$2"
            shift 2
            ;;
        --force_train)
            FORCE_TRAIN=true
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$INVERSION_OUT_DIR" ]]; then
    INVERSION_OUT_DIR="$OUT_DIR"
fi

IFS=',' read -r -a GPU_IDS <<< "$CUDA_DEVICES"
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "No CUDA devices provided."
    exit 1
fi

mkdir -p "$OUT_DIR/logs" "$INVERSION_OUT_DIR/logs"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$OUT_DIR/.cache}"

precompile_inversion_plugin() {
    local gpu_id="$1"
    local log_file="$INVERSION_OUT_DIR/logs/precompile_3dgptir.log"

    echo "[$(date '+%F %T')] Precompiling PTIR native plugin on CUDA_VISIBLE_DEVICES=$gpu_id"
    {
        echo "config=$INVERSION_CONFIG_NAME"
        echo "dataset=$DATASET_CONFIG"
        echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
        CUDA_VISIBLE_DEVICES="$gpu_id" python - <<PY
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import threedgrut.utils.misc  # registers project resolvers
from threedgptir_tracer.setup_threedgptir import setup_threedgptir

OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l], replace=True)

with initialize_config_dir(config_dir="$PWD/configs", version_base=None):
    conf = compose(config_name="$INVERSION_CONFIG_NAME", overrides=["dataset=$DATASET_CONFIG"])

setup_threedgptir(conf)
PY
    } > "$log_file" 2>&1
    echo "[$(date '+%F %T')] PTIR native plugin is ready"
}

find_latest_checkpoint() {
    local scene="$1"
    local scene_dir="$OUT_DIR/$scene"
    if [[ ! -d "$scene_dir" ]]; then
        return 1
    fi

    find "$scene_dir" -mindepth 2 -maxdepth 2 -name ckpt_last.pt -printf '%T@ %p\n' \
        | sort -nr \
        | awk 'NR == 1 {print $2}'
}

run_inversion() {
    local scene="$1"
    local gpu_id="$2"
    local initialization_path="$3"
    local scene_path="$DATA_ROOT/$scene"
    local log_file="$INVERSION_OUT_DIR/logs/inversion_${scene}.log"

    echo "[$(date '+%F %T')] Starting PTIR inversion scene=$scene on CUDA_VISIBLE_DEVICES=$gpu_id"
    {
        echo "scene=$scene"
        echo "cuda_device=$gpu_id"
        echo "config=$INVERSION_CONFIG_NAME"
        echo "dataset=$DATASET_CONFIG"
        echo "path=$scene_path"
        echo "initialization.path=$initialization_path"
        echo "out_dir=$INVERSION_OUT_DIR"
        echo "experiment_name=${scene}_inversion"
        printf 'inversion_extra_args=%q ' "${INVERSION_EXTRA_ARGS[@]}"
        echo
        nvidia-smi || true
        CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
            --config-name "$INVERSION_CONFIG_NAME" \
            "dataset=$DATASET_CONFIG" \
            "path=$scene_path" \
            "out_dir=$INVERSION_OUT_DIR" \
            "experiment_name=${scene}_inversion" \
            "initialization.path=$initialization_path" \
            "${INVERSION_EXTRA_ARGS[@]}"
    } > "$log_file" 2>&1
    echo "[$(date '+%F %T')] Finished PTIR inversion scene=$scene on CUDA_VISIBLE_DEVICES=$gpu_id"
}

run_scene() {
    local scene="$1"
    local gpu_id="$2"
    local scene_path="$DATA_ROOT/$scene"
    local log_file="$OUT_DIR/logs/train_${scene}.log"
    local checkpoint_path=""

    if [[ ! -d "$scene_path" ]]; then
        echo "[$(date '+%F %T')] Scene path does not exist, skipping: $scene_path"
        return 0
    fi

    checkpoint_path="$(find_latest_checkpoint "$scene" || true)"
    if [[ -n "$checkpoint_path" && "$FORCE_TRAIN" != true ]]; then
        echo "[$(date '+%F %T')] Found checkpoint for scene=$scene: $checkpoint_path"
        echo "[$(date '+%F %T')] Skipping training for scene=$scene"
    else
        echo "[$(date '+%F %T')] Starting training scene=$scene on CUDA_VISIBLE_DEVICES=$gpu_id"
        {
            echo "scene=$scene"
            echo "cuda_device=$gpu_id"
            echo "config=$CONFIG_NAME"
            echo "dataset=$DATASET_CONFIG"
            echo "path=$scene_path"
            echo "initialization.num_gaussians=$INITIALIZATION_NUM_GAUSSIANS"
            echo "initialization.xyz_max=$INITIALIZATION_XYZ_MAX"
            echo "initialization.xyz_min=$INITIALIZATION_XYZ_MIN"
            echo "out_dir=$OUT_DIR"
            echo "experiment_name=$scene"
            printf 'extra_args=%q ' "${EXTRA_ARGS[@]}"
            echo
            nvidia-smi || true
            CUDA_VISIBLE_DEVICES="$gpu_id" python train.py \
                --config-name "$CONFIG_NAME" \
                "dataset=$DATASET_CONFIG" \
                "path=$scene_path" \
                "initialization.num_gaussians=$INITIALIZATION_NUM_GAUSSIANS" \
                "initialization.xyz_max=$INITIALIZATION_XYZ_MAX" \
                "initialization.xyz_min=$INITIALIZATION_XYZ_MIN" \
                "out_dir=$OUT_DIR" \
                "experiment_name=$scene" \
                "${EXTRA_ARGS[@]}"
        } > "$log_file" 2>&1
        echo "[$(date '+%F %T')] Finished training scene=$scene on CUDA_VISIBLE_DEVICES=$gpu_id"
        checkpoint_path="$(find_latest_checkpoint "$scene" || true)"
    fi

    if [[ "$RUN_INVERSION" == true ]]; then
        if [[ -z "$checkpoint_path" ]]; then
            echo "[$(date '+%F %T')] No stage1 checkpoint found for scene=$scene; skipping PTIR inversion."
            return 1
        fi
        run_inversion "$scene" "$gpu_id" "$checkpoint_path"
    fi
}

if [[ "$RUN_INVERSION" == true ]]; then
    precompile_inversion_plugin "${GPU_IDS[0]}"
fi

active_jobs=0
for idx in "${!SCENES[@]}"; do
    scene="${SCENES[$idx]}"
    gpu_id="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"

    run_scene "$scene" "$gpu_id" &
    active_jobs=$((active_jobs + 1))

    if [[ $active_jobs -ge ${#GPU_IDS[@]} ]]; then
        wait -n
        active_jobs=$((active_jobs - 1))
    fi
done

wait
echo "All StanfordORB jobs finished. Logs: $OUT_DIR/logs"
