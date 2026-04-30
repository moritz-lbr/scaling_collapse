#!/usr/bin/env bash

################ SLURM HEADER ################

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

usage() {
  cat <<'EOF'
Usage: ./analyze_cov_submit.sh <JOB_DIR> [SNAPSHOT_STRIDE] [DELTA_T] [LAYERS_CSV] [FRAME_DURATION_MS]
EOF
}

# Example:
#   ./analyze_cov_submit.sh /path/to/logs/job-12818757
#   ./analyze_cov_submit.sh /path/to/logs/job-12818757 10 10
#   ./analyze_cov_submit.sh /path/to/logs/job-12818757 10 10 Dense_0,Dense_1
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for arg in "$@"; do
  if [[ "${arg}" == --* ]]; then
    echo "Unknown option: ${arg}"
    usage
    exit 1
  fi
done

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

N=$(ls -1 "${JOB_DIR}"/*/training_log.json 2>/dev/null | wc -l | tr -d ' ')

if [[ "${N}" -eq 0 ]]; then
  echo "No training logs found in ${JOB_DIR}."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/analyze_cov_script.sh" && -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/analyze_cov_script.sh" ]]; then
  SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
fi
if [[ ! -f "${SCRIPT_DIR}/analyze_cov_script.sh" && "${JOB_DIR}" == *"/experiments/"* ]]; then
  PROJECT_ROOT="${JOB_DIR%%/experiments/*}"
  if [[ -f "${PROJECT_ROOT}/analyze_cov_script.sh" ]]; then
    SCRIPT_DIR="$(cd "${PROJECT_ROOT}" && pwd)"
  fi
fi
if [[ ! -f "${SCRIPT_DIR}/analyze_cov_script.sh" ]]; then
  echo "Could not locate analyze_cov_script.sh from ${SCRIPT_DIR}."
  exit 1
fi
SOURCE_DIR="${SCRIPT_DIR}"

export JOB_DIR SNAPSHOT_STRIDE DELTA_T LAYERS_CSV FRAME_DURATION_MS N SOURCE_DIR

sbatch \
  --export=ALL \
  --array=1-"${N}" \
  "${SCRIPT_DIR}/analyze_cov_script.sh"
