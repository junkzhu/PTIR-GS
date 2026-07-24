#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_DEVICES="2,3"
STAGE1_OUT_DIR="outputs/synthetic4relight"
INVERSION_OUT_DIR="outputs/synthetic4relight_wo_prior"
SCENES=(airbaloons chair hotdog jugs)
RUN_RELIGHT=true
FORCE_TRAIN=false
EXTRA_BASE_ARGS=()
ADDITIONAL_INVERSION_ARGS=()

usage() {
    cat <<EOF
Usage: $0 [options] [-- extra args for benchmark/synthetic4relight.sh]

Runs no-prior Synthetic4Relight PTIR inversion with the tuned scene-wise
edge-aware smoothness settings.

Options:
  --cuda_device DEVICES      Comma-separated GPU ids. Default: $CUDA_DEVICES
  --out_dir PATH             Stage1 output/checkpoint root. Default: $STAGE1_OUT_DIR
  --inversion_out_dir PATH   PTIR output root. Default: $INVERSION_OUT_DIR
  --scenes "A B C"           Space-separated scenes. Default: ${SCENES[*]}
                             Accepts airbaloons or air_baloons.
  --with_relight             Run relighting after inversion. Default: off.
  --force_train              Forward --force_train to stage1 script.
  -h, --help                 Show this help.

Tuned no-prior smoothness:
  airbaloons: lambda_edge_aware_smoothness=0.095, scale=1.0
  chair:      lambda_edge_aware_smoothness=1.0,   scale=1.0
  hotdog:     lambda_edge_aware_smoothness=0.5,   scale=1.0
  jugs:       lambda_edge_aware_smoothness=0.2,   scale=1.0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda_device)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --out_dir)
            STAGE1_OUT_DIR="$2"
            shift 2
            ;;
        --inversion_out_dir)
            INVERSION_OUT_DIR="$2"
            shift 2
            ;;
        --scenes)
            read -r -a SCENES <<< "$2"
            shift 2
            ;;
        --with_relight)
            RUN_RELIGHT=true
            shift
            ;;
        --force_train)
            FORCE_TRAIN=true
            shift
            ;;
        --inversion_args)
            read -r -a parsed_inversion_args <<< "$2"
            ADDITIONAL_INVERSION_ARGS+=("${parsed_inversion_args[@]}")
            shift 2
            ;;
        --)
            shift
            EXTRA_BASE_ARGS+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_BASE_ARGS+=("$1")
            shift
            ;;
    esac
done

normalize_scene() {
    local scene="$1"
    case "$scene" in
        air_baloons)
            echo "airbaloons"
            ;;
        *)
            echo "$scene"
            ;;
    esac
}

smoothness_lambda_for_scene() {
    local scene="$1"
    case "$scene" in
        airbaloons)
            echo "0.095"
            ;;
        chair)
            echo "1.0"
            ;;
        hotdog)
            echo "0.5"
            ;;
        jugs)
            echo "0.2"
            ;;
        *)
            echo "Unknown scene '$scene'. Expected one of: airbaloons chair hotdog jugs." >&2
            return 1
            ;;
    esac
}

IFS=',' read -r -a GPU_IDS <<< "$CUDA_DEVICES"
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "No CUDA devices provided." >&2
    exit 1
fi

BASE_ARGS=(
    --out_dir "$STAGE1_OUT_DIR"
    --inversion_out_dir "$INVERSION_OUT_DIR"
)
if [[ "$RUN_RELIGHT" != true ]]; then
    BASE_ARGS+=(--no_relight)
fi
if [[ "$FORCE_TRAIN" == true ]]; then
    BASE_ARGS+=(--force_train)
fi

COMMON_INVERSION_ARGS=(
    "loss.use_albedo_prior_regularization=false"
    "loss.use_roughness_prior_regularization=false"
)

run_scene() {
    local scene="$1"
    local gpu_id="$2"
    local smoothness_lambda
    smoothness_lambda="$(smoothness_lambda_for_scene "$scene")"

    local inversion_args=(
        "loss.lambda_edge_aware_smoothness=$smoothness_lambda"
        "${COMMON_INVERSION_ARGS[@]}"
        "${ADDITIONAL_INVERSION_ARGS[@]}"
    )

    echo "[$(date '+%F %T')] scene=$scene gpu=$gpu_id lambda_edge_aware_smoothness=$smoothness_lambda no_prior=true"
    "$SCRIPT_DIR/synthetic4relight.sh" \
        --cuda_device "$gpu_id" \
        "${BASE_ARGS[@]}" \
        --scenes "$scene" \
        --inversion_args "${inversion_args[*]}" \
        "${EXTRA_BASE_ARGS[@]}"
}

cd "$REPO_ROOT"
mkdir -p "$INVERSION_OUT_DIR/logs"

pids=()
scene_index=0
for raw_scene in "${SCENES[@]}"; do
    scene="$(normalize_scene "$raw_scene")"
    gpu_id="${GPU_IDS[$((scene_index % ${#GPU_IDS[@]}))]}"
    run_scene "$scene" "$gpu_id" &
    pids+=("$!")
    scene_index=$((scene_index + 1))

    if [[ ${#pids[@]} -ge ${#GPU_IDS[@]} ]]; then
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        pids=()
    fi
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "[$(date '+%F %T')] Synthetic4Relight no-prior tuned runs finished. Outputs: $INVERSION_OUT_DIR"
