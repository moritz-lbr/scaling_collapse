#!/bin/bash

################ SLURM HEADER ################
#SBATCH --job-name=output_cov_analysis
#SBATCH --mail-type=ALL,ARRAY_TASKS

#SBATCH --mem=128G
#SBATCH --time=12:00:00

#SBATCH --partition=inter,cluster,th-ws,cip,cip-ws

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

: "${JOB_DIR:?JOB_DIR must be provided via output_analyze_cov_submit.sh}"
: "${OUTPUT_SAMPLE_INDEX:=1}"

if [[ -n "${SOURCE_DIR:-}" ]]; then
  SLURMSCRIPTDIR="$(cd "${SOURCE_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/training_analysis/corr_analysis_output.py" ]]; then
  SLURMSCRIPTDIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
elif [[ "${JOB_DIR}" == *"/experiments/"* && -f "${JOB_DIR%%/experiments/*}/training_analysis/corr_analysis_output.py" ]]; then
  SLURMSCRIPTDIR="$(cd "${JOB_DIR%%/experiments/*}" && pwd)"
else
  SLURMSCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
SOURCEDIR="${SLURMSCRIPTDIR}"
CORR_OUTPUT_PROGRAM="training_analysis/corr_analysis_output.py"
COV_PROGRAM="visualizations/cov.py"
XJ_PROGRAM="visualizations/plot_xj_terms.py"
SLURMOUTDIR="${JOB_DIR}/slurm_output_output_cov_analysis"

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
value = cfg.get("training", {}).get("save_loss_frequency")

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
LOGIT_COUNT="$(
  pixi run python - "${SIM_CONFIG}" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1])) or {}
network = cfg.get("network") or cfg.get("simulation_parameters", {}).get("network", {}) or {}
training = cfg.get("training") or cfg.get("simulation_parameters", {}).get("training", {}) or {}

def as_int(value):
    try:
        return int(value)
    except Exception:
        return None

nodes_per_layer = network.get("nodes_per_layer") or {}
dense_1 = as_int(nodes_per_layer.get("Dense_1"))
output_dim = as_int(training.get("training_data", {}).get("output_dimension"))
dense_layer_count = sum(str(key).startswith("Dense_") for key in nodes_per_layer)

value = dense_1 if dense_1 is not None and dense_layer_count <= 2 else output_dim
if value is None:
    value = dense_1
if value is None:
    value = as_int((network.get("base_layer_width") or {}).get("Dense_1"))
if value is None:
    value = as_int((network.get("base_layer_widths") or {}).get("Dense_1"))

print(value if value is not None else 1)
PY
)"

if ! [[ "${OUTPUT_SAMPLE_INDEX}" =~ ^[0-9]+$ ]] || (( OUTPUT_SAMPLE_INDEX < 1 )); then
  echo "OUTPUT_SAMPLE_INDEX must be a positive integer, got: ${OUTPUT_SAMPLE_INDEX}"
  exit 1
fi
if ! [[ "${LOGIT_COUNT}" =~ ^[0-9]+$ ]] || (( LOGIT_COUNT < 1 )); then
  echo "Could not infer a positive logit count from ${SIM_CONFIG}; got: ${LOGIT_COUNT}"
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

logits_figure_dir() {
  local logit_index="$1"
  printf '%s\n' "${COV_ROOT}/${TASK_NAME}/${JOB_ID}/${TRAINING_NAME}/logits/sample_${OUTPUT_SAMPLE_INDEX}/logit_${logit_index}"
}

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
  echo "@output_sample_idx ${OUTPUT_SAMPLE_INDEX}"
  echo "@logit_count       ${LOGIT_COUNT}"
  echo "@snapshot_stride   ${SNAPSHOT_STRIDE}"
  echo "@frame_duration_ms ${FRAME_DURATION_MS}"
}

run_corr_output_analysis() {
  local logit_index="$1"
  pixi run python "${CORR_OUTPUT_PROGRAM}" \
    --log-dir "${SELECTED_LOG}" \
    --delta-t "${DELTA_T}" \
    --sample-index "${OUTPUT_SAMPLE_INDEX}" \
    --logit-index "${logit_index}" \
    --output "${OUTPUTDIR}"
}

run_logits_cov_visualization() {
  local logit_index="$1"
  cov_file="${OUTPUTDIR}/cov_logits_sample_${OUTPUT_SAMPLE_INDEX}_logit_${logit_index}.npy"
  if [[ ! -f "${cov_file}" ]]; then
    echo "Expected logits covariance file missing: ${cov_file}"
    exit 1
  fi

  logits_output_dir="$(logits_figure_dir "${logit_index}")"

  pixi run python "${COV_PROGRAM}" \
    --cov-path "${cov_file}" \
    --delta-t "${DELTA_T}" \
    --output-dir "${logits_output_dir}" \
    --gif-name "corr_logits_sample_${OUTPUT_SAMPLE_INDEX}_logit_${logit_index}_log.gif" \
    --frame-duration-ms "${FRAME_DURATION_MS}" \
    --frame-stride "${SNAPSHOT_STRIDE}" \
    --save-loss-frequency "${SAVE_LOSS_FREQUENCY}"
}

run_logits_xj_terms_visualization() {
  local logit_index="$1"
  xj_file="${OUTPUTDIR}/xj_logits_sample_${OUTPUT_SAMPLE_INDEX}_logit_${logit_index}.npy"
  if [[ ! -f "${xj_file}" ]]; then
    echo "Expected logits xj file missing: ${xj_file}"
    exit 1
  fi

  logits_output_dir="$(logits_figure_dir "${logit_index}")"

  pixi run python "${XJ_PROGRAM}" \
    --xj-path "${xj_file}" \
    --output "${logits_output_dir}/xj_terms.png" \
    --save-loss-frequency "${SAVE_LOSS_FREQUENCY}" \
    --layer "logits_logit_${logit_index}"
}


print_info

for (( LOGIT_INDEX = 1; LOGIT_INDEX <= LOGIT_COUNT; LOGIT_INDEX++ )); do
  echo "# Logit ${LOGIT_INDEX}/${LOGIT_COUNT}"

  if ! run_corr_output_analysis "${LOGIT_INDEX}"; then
    echo "corr_analysis_output failed for logit ${LOGIT_INDEX}"
    exit 1
  fi

  if ! run_logits_cov_visualization "${LOGIT_INDEX}"; then
    echo "logits cov visualization failed for logit ${LOGIT_INDEX}"
    exit 1
  fi

  if ! run_logits_xj_terms_visualization "${LOGIT_INDEX}"; then
    echo "logits xj terms visualization failed for logit ${LOGIT_INDEX}"
    exit 1
  fi
done

echo "Output covariance analysis and visualizations completed successfully."
exit 0
