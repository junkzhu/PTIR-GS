#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_DEVICES="0,1,2,3"
STAGE1_OUT_DIR="outputs/rt4relight"
INVERSION_OUT_DIR="outputs/rt4relight_wo_prior"
SCENES=(barrels bread plastica teacup)
RUN_RELIGHT=true
FORCE_TRAIN=false
EXTRA_BASE_ARGS=()
declare -A SMOOTHNESS_LAMBDAS=(
    [barrels]="0.095"
    [bread]="1.0"
    [plastica]="2.0"
    [teacup]="1.0"
)

usage() {
    cat <<EOF
Usage: $0 [options] [-- extra args for benchmark/rt4relight.sh]

Runs no-prior RT4Relight PTIR inversion with tuned scene-wise edge-aware
material smoothness. Albedo and roughness priors are disabled.

Options:
  --cuda_device DEVICES      Comma-separated GPU ids. Default: $CUDA_DEVICES
  --out_dir PATH             Stage1 output/checkpoint root. Default: $STAGE1_OUT_DIR
  --inversion_out_dir PATH   PTIR output root. Default: $INVERSION_OUT_DIR
  --scenes "A B C"           Space-separated scenes. Default: ${SCENES[*]}
  --smoothness "S=V ..."     Override scene smoothness, e.g. "teacup=0.5 bread=0.1".
  --with_relight             Run relighting after inversion. Default: off.
  --force_train              Forward --force_train to the stage1 script.
  -h, --help                 Show this help.

Tuned no-prior smoothness:
  barrels:  iterations=800, lambda_edge_aware_smoothness=0.095
  bread:    iterations=800, lambda_edge_aware_smoothness=1.0
  plastica: iterations=800, lambda_edge_aware_smoothness=2.0
  teacup:   iterations=800, lambda_edge_aware_smoothness=1.0
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
        --smoothness)
            read -r -a smoothness_overrides <<< "$2"
            for override in "${smoothness_overrides[@]}"; do
                scene="${override%%=*}"
                value="${override#*=}"
                if [[ "$scene" == "$override" || -z "${SMOOTHNESS_LAMBDAS[$scene]+x}" ]]; then
                    echo "Invalid smoothness override '$override'." >&2
                    exit 1
                fi
                SMOOTHNESS_LAMBDAS[$scene]="$value"
            done
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

smoothness_lambda_for_scene() {
    local scene="$1"
    if [[ -z "${SMOOTHNESS_LAMBDAS[$scene]+x}" ]]; then
        echo "Unknown scene '$scene'. Expected one of: barrels bread plastica teacup." >&2
        return 1
    fi
    echo "${SMOOTHNESS_LAMBDAS[$scene]}"
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
        "n_iterations=800"
        "loss.lambda_edge_aware_smoothness=$smoothness_lambda"
        "${COMMON_INVERSION_ARGS[@]}"
    )

    echo "[$(date '+%F %T')] scene=$scene gpu=$gpu_id n_iterations=800 lambda_edge_aware_smoothness=$smoothness_lambda no_prior=true"
    "$SCRIPT_DIR/rt4relight.sh" \
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
for scene in "${SCENES[@]}"; do
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

echo "[$(date '+%F %T')] RT4Relight no-prior tuned runs finished. Outputs: $INVERSION_OUT_DIR"
