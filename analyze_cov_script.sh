#!/bin/bash

################ SLURM HEADER ################
#SBATCH --job-name=cov_analysis
#SBATCH --mail-type=ALL,ARRAY_TASKS

#SBATCH --mem=128G
#SBATCH --time=12:00:00

#SBATCH --partition=inter,cluster,th-ws,cip,cip-ws

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

: "${JOB_DIR:?JOB_DIR must be provided via analyze_cov_submit.sh}"
SNAPSHOT_STRIDE="${SNAPSHOT_STRIDE:-1}"
LAYERS_CSV="${LAYERS_CSV:-Dense_0}"
FRAME_DURATION_MS="${FRAME_DURATION_MS:-80}"

SLURMSCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCEDIR="${SLURMSCRIPTDIR}"
CORR_PROGRAM="training_analysis/corr_analysis.py"
COV_PROGRAM="visualizations/cov.py"
SLURMOUTDIR="${JOB_DIR}/slurm_output_cov_analysis"

mkdir -p "${SLURMOUTDIR}"
exec > "${SLURMOUTDIR}/task-${SLURM_ARRAY_TASK_ID}.out" 2>&1

mapfile -t JOBS < <(
    ls -1 "${JOB_DIR}"/*/training_log.json 2>/dev/null |
    sort -V
)

TOTAL=${#JOBS[@]}
IDX=$((SLURM_ARRAY_TASK_ID - 1))

if (( TOTAL == 0 )); then
  echo "No training logs found in ${JOB_DIR}."
  exit 1
fi
if (( IDX < 0 || IDX >= TOTAL )); then
  echo "Task index ${SLURM_ARRAY_TASK_ID} is out of range (have ${TOTAL} trainings). Exiting gracefully."
  exit 0
fi

SELECTED_LOG="${JOBS[$IDX]}"
TRAINING_DIR="${SELECTED_LOG%/*}"
TRAINING_NAME="$(basename "${TRAINING_DIR}")"
JOB_ID="$(basename "$(dirname "${TRAINING_DIR}")")"
OUTPUTDIR="${TRAINING_DIR}/weight_metrics"
SIM_CONFIG="${TRAINING_DIR}/simulation_config.yaml"

mkdir -p "${OUTPUTDIR}"

TASK_NAME="$(pixi run python -c "import os,sys,yaml; cfg=yaml.safe_load(open(sys.argv[1])) or {}; task=cfg.get('training',{}).get('training_data',{}).get('task',''); print(os.path.basename(str(task).rstrip('/')) or 'unknown_task')" "${SIM_CONFIG}")"
SAVE_LOSS_FREQUENCY="$(
  pixi run python - "${SIM_CONFIG}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1])) or {}
value = cfg.get("training", {}).get("save_loss_frequency", 1)

if isinstance(value, str):
    if value.strip().lower() == "epoch":
        print(1)
    else:
        try:
            print(int(float(value)))
        except Exception:
            print(1)
else:
    try:
        print(int(value))
    except Exception:
        print(1)
PY
)"

IFS=',' read -r -a RAW_LAYERS <<< "${LAYERS_CSV}"
LAYERS=()
for layer in "${RAW_LAYERS[@]}"; do
  trimmed="$(echo "${layer}" | xargs)"
  if [[ -n "${trimmed}" ]]; then
    LAYERS+=("${trimmed}")
  fi
done

if (( ${#LAYERS[@]} == 0 )); then
  echo "No valid layers parsed from LAYERS_CSV='${LAYERS_CSV}'."
  exit 1
fi

# Derive project root from JOB_DIR and write figures there.
# Example JOB_DIR: /.../scaling_collapse/experiments/test/logs/job-123456
if [[ "${JOB_DIR}" == *"/experiments/"* ]]; then
  PROJECT_ROOT="${JOB_DIR%%/experiments/*}"
else
  PROJECT_ROOT="$(pwd)"
fi
COV_ROOT="${PROJECT_ROOT}/figures_cov"

print_info() {
  echo "# Job info"
  echo "@job_name          ${SLURM_JOB_NAME}"
  echo "@job_id            ${SLURM_JOB_ID}"
  echo "@array_job_id      ${SLURM_ARRAY_JOB_ID}"
  echo "@array_task_id     ${SLURM_ARRAY_TASK_ID}"
  echo "@start_date        $(date)"
  echo "@host              ${SLURMD_NODENAME}"
  echo "@selected_log      ${SELECTED_LOG}"
  echo "@training_dir      ${TRAINING_DIR}"
  echo "@task_name         ${TASK_NAME}"
  echo "@save_loss_freq    ${SAVE_LOSS_FREQUENCY}"
  echo "@layers            ${LAYERS[*]}"
  echo "@snapshot_stride   ${SNAPSHOT_STRIDE}"
  echo "@frame_duration_ms ${FRAME_DURATION_MS}"
}

run_corr_analysis() {
  pixi run python "${CORR_PROGRAM}" \
    --log-dir "${SELECTED_LOG}" \
    --output "${OUTPUTDIR}" \
    --layers "${LAYERS[@]}"
}

run_cov_visualization() {
  for layer in "${LAYERS[@]}"; do
    cov_file="${OUTPUTDIR}/cov_${layer}.npy"
    if [[ ! -f "${cov_file}" ]]; then
      echo "Expected covariance file missing: ${cov_file}"
      exit 1
    fi

    layer_output_dir="${COV_ROOT}/${TASK_NAME}/${JOB_ID}/${TRAINING_NAME}/${layer}"

    pixi run python "${COV_PROGRAM}" \
      --cov-path "${cov_file}" \
      --output-dir "${layer_output_dir}" \
      --gif-name "cov_${layer}.gif" \
      --frame-duration-ms "${FRAME_DURATION_MS}" \
      --frame-stride "${SNAPSHOT_STRIDE}" \
      --save-loss-frequency "${SAVE_LOSS_FREQUENCY}"
  done
}

print_info

run_corr_analysis
corr_returncode=$?
if [[ ${corr_returncode} -ne 0 ]]; then
  echo "corr_analysis failed with code ${corr_returncode}"
  exit ${corr_returncode}
fi

run_cov_visualization
cov_returncode=$?
if [[ ${cov_returncode} -ne 0 ]]; then
  echo "cov visualization failed with code ${cov_returncode}"
  exit ${cov_returncode}
fi

echo "Analysis and covariance visualizations completed successfully."
exit 0
