#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INVERSION_OUT_DIR="outputs/tensoir_wo_prior"
FORWARDED_ARGS=()
ADDITIONAL_INVERSION_ARGS=()

usage() {
    cat <<EOF
Usage: $0 [options] [-- extra hydra args]

Runs benchmark/tensoir.sh with albedo and roughness prior regularization
disabled during PTIR inversion. Stage-1 checkpoints still default to
outputs/tensoir, while PTIR outputs default to $INVERSION_OUT_DIR.

Options:
  --inversion_out_dir PATH   PTIR output root. Default: $INVERSION_OUT_DIR
  --inversion_args "ARGS"    Additional PTIR Hydra overrides.
  -h, --help                 Show this help.

All other options are forwarded unchanged to benchmark/tensoir.sh.

Examples:
  $0 --cuda_device 0,1,2,3
  $0 --cuda_device 0 --scenes "lego"
  $0 --cuda_device 0,1 --inversion_args "n_iterations=400"
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --inversion_out_dir)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --inversion_out_dir" >&2
                exit 1
            fi
            INVERSION_OUT_DIR="$2"
            shift 2
            ;;
        --inversion_args)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --inversion_args" >&2
                exit 1
            fi
            read -r -a parsed_inversion_args <<< "$2"
            ADDITIONAL_INVERSION_ARGS+=("${parsed_inversion_args[@]}")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

# Keep these overrides last so this script cannot accidentally enable either
# material prior through --inversion_args.
INVERSION_ARGS=(
    "${ADDITIONAL_INVERSION_ARGS[@]}"
    "loss.use_albedo_prior_regularization=false"
    "loss.use_roughness_prior_regularization=false"
)

cd "$REPO_ROOT"
exec "$SCRIPT_DIR/tensoir.sh" \
    --inversion_out_dir "$INVERSION_OUT_DIR" \
    --inversion_args "${INVERSION_ARGS[*]}" \
    "${FORWARDED_ARGS[@]}"
