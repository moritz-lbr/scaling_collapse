#!/usr/bin/env bash

################ SLURM HEADER ################

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

# Example:
#   ./analyze_cov_submit.sh /path/to/logs/job-12818757
#   ./analyze_cov_submit.sh /path/to/logs/job-12818757 2
#   ./analyze_cov_submit.sh /path/to/logs/job-12818757 3 Dense_0,Dense_1
JOB_DIR=${1:? "Usage: ./analyze_cov_submit.sh <JOB_DIR> [SNAPSHOT_STRIDE] [DELTA_T] [LAYERS_CSV] [FRAME_DURATION_MS]"}
SNAPSHOT_STRIDE=${2:-10}
DELTA_T=${3:-10}
LAYERS_CSV=${4:-Dense_0}
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

sbatch \
  --export=ALL,JOB_DIR="${JOB_DIR}",SNAPSHOT_STRIDE="${SNAPSHOT_STRIDE}",DELTA_T="${DELTA_T}",LAYERS_CSV="${LAYERS_CSV}",FRAME_DURATION_MS="${FRAME_DURATION_MS}",N="${N}" \
  --array=1-"${N}" \
  analyze_cov_script.sh
