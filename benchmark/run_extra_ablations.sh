#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CUDA_DEVICES=(0 1 2 3)
SCENES=(lego armadillo hotdog ficus)
SPP_VALUES=(1 8 16 32 64)
BOUNCE_VALUES=(2 3 5 8 10)

OUT_DIR="outputs/tensoir"
SPP_INVERSION_OUT_ROOT="outputs/tensoir_inversion_spp"
BOUNCE_INVERSION_OUT_ROOT="outputs/tensoir_max_bounces"
RENDER_FRAME_STRIDE=5
FIXED_EVAL_MAX_BOUNCES=10
DATASET_CONFIG="tensoir"
INVERSION_CONFIG_NAME="inversions/nerf_synthetic_3dgptir.yaml"
INVERSION_EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --no_material_priors   Disable albedo and roughness prior regularization for all PTIR inversion tasks.
  --no_albedo_prior      Disable albedo prior regularization for all PTIR inversion tasks.
  --no_roughness_prior   Disable roughness prior regularization for all PTIR inversion tasks.
  --inversion_args "ARGS" Extra Hydra args appended to every PTIR inversion task.
  --eval_max_bounces N   Fixed render.max_bounces for post-inversion render/relight metrics. Default: $FIXED_EVAL_MAX_BOUNCES
  -h, --help             Show this help.

Example:
  $0 --no_material_priors
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no_material_priors)
            INVERSION_EXTRA_ARGS+=(
                "loss.use_albedo_prior_regularization=false"
                "loss.use_roughness_prior_regularization=false"
            )
            shift
            ;;
        --no_albedo_prior)
            INVERSION_EXTRA_ARGS+=("loss.use_albedo_prior_regularization=false")
            shift
            ;;
        --no_roughness_prior)
            INVERSION_EXTRA_ARGS+=("loss.use_roughness_prior_regularization=false")
            shift
            ;;
        --inversion_args)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --inversion_args"
                exit 1
            fi
            read -r -a parsed_inversion_args <<< "$2"
            INVERSION_EXTRA_ARGS+=("${parsed_inversion_args[@]}")
            shift 2
            ;;
        --eval_max_bounces)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --eval_max_bounces"
                exit 1
            fi
            FIXED_EVAL_MAX_BOUNCES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

QUEUE_DIR="$OUT_DIR/logs/run_all_$(date '+%Y%m%d_%H%M%S')"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$OUT_DIR/.cache}"

mkdir -p "$QUEUE_DIR" "$OUT_DIR/logs"

join_by_comma() {
    local IFS=,
    echo "$*"
}

make_inversion_override() {
    local base_override="$1"

    if [[ ${#INVERSION_EXTRA_ARGS[@]} -eq 0 ]]; then
        printf '%s' "$base_override"
    else
        printf '%s %s' "$base_override" "${INVERSION_EXTRA_ARGS[*]}"
    fi
}

precompile_inversion_plugin() {
    local gpu_id="$1"
    local log_file="$QUEUE_DIR/precompile_3dgptir.log"

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

get_task() {
    local queue_file="$1"
    local lock_file="$2"
    local line

    {
        flock -x 200
        if ! IFS= read -r line < "$queue_file"; then
            return 1
        fi
        tail -n +2 "$queue_file" > "${queue_file}.tmp"
        mv "${queue_file}.tmp" "$queue_file"
    } 200>"$lock_file"

    printf '%s\n' "$line"
}

run_worker() {
    local gpu_id="$1"
    local queue_file="$2"
    local lock_file="$3"
    local queue_name="$4"
    local task_line task_type label scene inversion_out_dir override

    while task_line="$(get_task "$queue_file" "$lock_file")"; do
        IFS=$'\t' read -r task_type label scene inversion_out_dir override <<< "$task_line"
        echo "[$(date '+%F %T')] GPU $gpu_id starting $queue_name task: type=$task_type label=$label scene=$scene"

        if [[ "$task_type" == "stage1" ]]; then
            bash benchmark/tensoir.sh \
                --cuda_device "$gpu_id" \
                --out_dir "$OUT_DIR" \
                --scenes "$scene" \
                --render_frame_stride "$RENDER_FRAME_STRIDE" \
                --no_inversion
        else
            bash benchmark/tensoir.sh \
                --cuda_device "$gpu_id" \
                --out_dir "$OUT_DIR" \
                --inversion_out_dir "$inversion_out_dir" \
                --scenes "$scene" \
                --render_frame_stride "$RENDER_FRAME_STRIDE" \
                --no_precompile \
                --render_args "--override render.max_bounces=$FIXED_EVAL_MAX_BOUNCES" \
                --inversion_args "$override"
        fi

        echo "[$(date '+%F %T')] GPU $gpu_id finished $queue_name task: type=$task_type label=$label scene=$scene"
    done
}

run_queue() {
    local queue_name="$1"
    local queue_file="$2"
    local lock_file="$queue_file.lock"
    local pids=()
    local status=0

    echo "[$(date '+%F %T')] Starting queue: $queue_name"
    for gpu_id in "${CUDA_DEVICES[@]}"; do
        run_worker "$gpu_id" "$queue_file" "$lock_file" "$queue_name" &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done

    if [[ "$status" -ne 0 ]]; then
        echo "[$(date '+%F %T')] Queue failed: $queue_name"
        exit "$status"
    fi
    echo "[$(date '+%F %T')] Finished queue: $queue_name"
}

stage1_queue="$QUEUE_DIR/stage1.tsv"
: > "$stage1_queue"
for scene in "${SCENES[@]}"; do
    printf 'stage1\tdefault\t%s\t\t\n' "$scene" >> "$stage1_queue"
done

ablation_queue="$QUEUE_DIR/ablation.tsv"
: > "$ablation_queue"
for spp in "${SPP_VALUES[@]}"; do
    for scene in "${SCENES[@]}"; do
        override="$(make_inversion_override "render.inversion_spp=$spp test_last=false")"
        printf 'spp\t%s\t%s\t%s%s\t%s\n' \
            "$spp" "$scene" "$SPP_INVERSION_OUT_ROOT" "$spp" "$override" >> "$ablation_queue"
    done
done
for max_bounces in "${BOUNCE_VALUES[@]}"; do
    for scene in "${SCENES[@]}"; do
        override="$(make_inversion_override "render.max_bounces=$max_bounces test_last=false")"
        printf 'max_bounces\t%s\t%s\t%s%s\t%s\n' \
            "$max_bounces" "$scene" "$BOUNCE_INVERSION_OUT_ROOT" "$max_bounces" "$override" >> "$ablation_queue"
    done
done

echo "[$(date '+%F %T')] GPUs: $(join_by_comma "${CUDA_DEVICES[@]}")"
echo "[$(date '+%F %T')] Queue files: $QUEUE_DIR"
echo "[$(date '+%F %T')] Fixed eval render.max_bounces: $FIXED_EVAL_MAX_BOUNCES"
if [[ ${#INVERSION_EXTRA_ARGS[@]} -gt 0 ]]; then
    printf "[%s] Extra inversion args:" "$(date '+%F %T')"
    printf " %q" "${INVERSION_EXTRA_ARGS[@]}"
    echo
fi

run_queue "stage1" "$stage1_queue"
precompile_inversion_plugin "${CUDA_DEVICES[0]}"
run_queue "ablation" "$ablation_queue"

echo "[$(date '+%F %T')] All TensoIR ablation tasks finished."
