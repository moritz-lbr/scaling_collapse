#!/usr/bin/env bash

################ SLURM HEADER ################

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

usage() {
  cat <<'EOF'
Usage: ./analyze_cov_submit.sh <JOB_DIR> [SNAPSHOT_STRIDE] [DELTA_T] [LAYERS_CSV] [FRAME_DURATION_MS] [--output_sample_index N]

Options:
  --output_sample_index N   1-based sample index for the logits/output-term covariance analysis (default: 1)
EOF
}

# Example:
#   ./analyze_cov_submit.sh /path/to/logs/job-12818757
#   ./analyze_cov_submit.sh /path/to/logs/job-12818757 10 10
#   ./analyze_cov_submit.sh /path/to/logs/job-12818757 10 10 Dense_0,Dense_1
OUTPUT_SAMPLE_INDEX=1
POSITIONAL_ARGS=()

while (($# > 0)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --output_sample_index)
      if (($# < 2)); then
        echo "Missing value for --output_sample_index"
        exit 1
      fi
      OUTPUT_SAMPLE_INDEX="$2"
      shift 2
      ;;
    --output_sample_index=*)
      OUTPUT_SAMPLE_INDEX="${1#*=}"
      shift
      ;;
    --*)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

JOB_DIR=${1:-}
if [[ -z "${JOB_DIR}" ]]; then
  usage
  exit 1
fi

SNAPSHOT_STRIDE=${2:-10}
DELTA_T=${3:-10}
LAYERS_CSV=${4:-Dense_0,Dense_1}
FRAME_DURATION_MS=${5:-80}

if ! [[ "${SNAPSHOT_STRIDE}" =~ ^[0-9]+$ ]] || (( SNAPSHOT_STRIDE < 1 )); then
  echo "SNAPSHOT_STRIDE must be a positive integer, got: ${SNAPSHOT_STRIDE}"
  exit 1
fi

if ! [[ "${FRAME_DURATION_MS}" =~ ^[0-9]+$ ]] || (( FRAME_DURATION_MS < 1 )); then
  echo "FRAME_DURATION_MS must be a positive integer, got: ${FRAME_DURATION_MS}"
  exit 1
fi

if ! [[ "${OUTPUT_SAMPLE_INDEX}" =~ ^[0-9]+$ ]] || (( OUTPUT_SAMPLE_INDEX < 1 )); then
  echo "OUTPUT_SAMPLE_INDEX must be a positive integer, got: ${OUTPUT_SAMPLE_INDEX}"
  exit 1
fi

N=$(ls -1 "${JOB_DIR}"/*/training_log.json 2>/dev/null | wc -l | tr -d ' ')

if [[ "${N}" -eq 0 ]]; then
  echo "No training logs found in ${JOB_DIR}."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export JOB_DIR SNAPSHOT_STRIDE DELTA_T LAYERS_CSV FRAME_DURATION_MS OUTPUT_SAMPLE_INDEX N

sbatch \
  --export=ALL \
  --array=1-"${N}" \
  analyze_cov_script.sh
