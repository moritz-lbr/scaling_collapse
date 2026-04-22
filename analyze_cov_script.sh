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
: "${OUTPUT_SAMPLE_INDEX:=1}"

SLURMSCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCEDIR="${SLURMSCRIPTDIR}"
CORR_PROGRAM="training_analysis/corr_analysis.py"
CORR_OUTPUT_PROGRAM="training_analysis/corr_analysis_output.py"
COV_PROGRAM="visualizations/cov.py"
XJ_PROGRAM="visualizations/plot_xj_terms.py"
COMBINED_FIGURE_PROGRAM="visualizations/combine_corr_with_training_window.py"
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

if ! [[ "${OUTPUT_SAMPLE_INDEX}" =~ ^[0-9]+$ ]] || (( OUTPUT_SAMPLE_INDEX < 1 )); then
  echo "OUTPUT_SAMPLE_INDEX must be a positive integer, got: ${OUTPUT_SAMPLE_INDEX}"
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

layer_figure_dir() {
  local layer="$1"
  printf '%s\n' "${COV_ROOT}/${TASK_NAME}/${JOB_ID}/${TRAINING_NAME}/${layer}"
}

logits_figure_dir() {
  printf '%s\n' "${COV_ROOT}/${TASK_NAME}/${JOB_ID}/${TRAINING_NAME}/logits/sample_${OUTPUT_SAMPLE_INDEX}"
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
  echo "@layers            ${LAYERS[*]}"
  echo "@output_sample_idx ${OUTPUT_SAMPLE_INDEX}"
  echo "@snapshot_stride   ${SNAPSHOT_STRIDE}"
  echo "@frame_duration_ms ${FRAME_DURATION_MS}"
}

run_corr_analysis() {
  pixi run python "${CORR_PROGRAM}" \
    --log-dir "${SELECTED_LOG}" \
    --delta-t "${DELTA_T}" \
    --output "${OUTPUTDIR}" \
    --layers "${LAYERS[@]}"
}

run_corr_output_analysis() {
  pixi run python "${CORR_OUTPUT_PROGRAM}" \
    --log-dir "${SELECTED_LOG}" \
    --delta-t "${DELTA_T}" \
    --sample-index "${OUTPUT_SAMPLE_INDEX}" \
    --output "${OUTPUTDIR}"
}

run_cov_visualization() {
  for layer in "${LAYERS[@]}"; do
    cov_file="${OUTPUTDIR}/cov_${layer}.npy"
    if [[ ! -f "${cov_file}" ]]; then
      echo "Expected covariance file missing: ${cov_file}"
      exit 1
    fi

    layer_output_dir="$(layer_figure_dir "${layer}")"

    pixi run python "${COV_PROGRAM}" \
      --cov-path "${cov_file}" \
      --delta-t "${DELTA_T}" \
      --output-dir "${layer_output_dir}" \
      --gif-name "corr_${layer}_log.gif" \
      --frame-duration-ms "${FRAME_DURATION_MS}" \
      --frame-stride "${SNAPSHOT_STRIDE}" \
      --save-loss-frequency "${SAVE_LOSS_FREQUENCY}"
  done
}

run_logits_cov_visualization() {
  cov_file="${OUTPUTDIR}/cov_logits_sample_${OUTPUT_SAMPLE_INDEX}.npy"
  if [[ ! -f "${cov_file}" ]]; then
    echo "Expected logits covariance file missing: ${cov_file}"
    exit 1
  fi

  logits_output_dir="$(logits_figure_dir)"

  pixi run python "${COV_PROGRAM}" \
    --cov-path "${cov_file}" \
    --delta-t "${DELTA_T}" \
    --output-dir "${logits_output_dir}" \
    --gif-name "corr_logits_sample_${OUTPUT_SAMPLE_INDEX}_log.gif" \
    --frame-duration-ms "${FRAME_DURATION_MS}" \
    --frame-stride "${SNAPSHOT_STRIDE}" \
    --save-loss-frequency "${SAVE_LOSS_FREQUENCY}"
}

run_xj_terms_visualization() {
  for layer in "${LAYERS[@]}"; do
    xj_file="${OUTPUTDIR}/xj_${layer}.npy"
    if [[ ! -f "${xj_file}" ]]; then
      echo "Expected xj file missing: ${xj_file}"
      exit 1
    fi

    layer_output_dir="$(layer_figure_dir "${layer}")"

    pixi run python "${XJ_PROGRAM}" \
      --xj-path "${xj_file}" \
      --output "${layer_output_dir}/xj_terms.png" \
      --save-loss-frequency "${SAVE_LOSS_FREQUENCY}" \
      --title "${layer}"
  done
}

run_logits_xj_terms_visualization() {
  xj_file="${OUTPUTDIR}/xj_logits_sample_${OUTPUT_SAMPLE_INDEX}.npy"
  if [[ ! -f "${xj_file}" ]]; then
    echo "Expected logits xj file missing: ${xj_file}"
    exit 1
  fi

  logits_output_dir="$(logits_figure_dir)"

  pixi run python "${XJ_PROGRAM}" \
    --xj-path "${xj_file}" \
    --output "${logits_output_dir}/xj_terms.png" \
    --save-loss-frequency "${SAVE_LOSS_FREQUENCY}" \
    --layer "logits"
}

run_combined_figure_visualization() {
  for layer in "${LAYERS[@]}"; do
    cov_dir="$(layer_figure_dir "${layer}")"
    if [[ ! -d "${cov_dir}" ]]; then
      echo "Expected covariance directory missing: ${cov_dir}"
      exit 1
    fi

    metrics_image="figures_weight_update_similarity/${TASK_NAME}/${JOB_ID}/weight_metrics_${layer}.png"
    if [[ ! -f "${metrics_image}" ]]; then
      echo "Skipping combined figure for ${layer}: metrics image missing at ${metrics_image}"
      continue
    fi

    pixi run python "${COMBINED_FIGURE_PROGRAM}" \
      --corr-dir "${cov_dir}/cov_log_frames" \
      --metrics-image "${metrics_image}" \
      --job-dir "${JOB_DIR}" \
      --output-gif "${cov_dir}/combined_weight_metrics_${layer}.gif" \
      --keep-frames-dir "${cov_dir}/combined_weight_metrics_${layer}_frames" \
      --sampling-mode log-decades
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

run_xj_terms_visualization
xj_returncode=$?
if [[ ${xj_returncode} -ne 0 ]]; then
  echo "xj terms visualization failed with code ${xj_returncode}"
  exit ${xj_returncode}
fi

run_corr_output_analysis
corr_output_returncode=$?
if [[ ${corr_output_returncode} -ne 0 ]]; then
  echo "corr_analysis_output failed with code ${corr_output_returncode}"
  exit ${corr_output_returncode}
fi

run_logits_cov_visualization
logits_cov_returncode=$?
if [[ ${logits_cov_returncode} -ne 0 ]]; then
  echo "logits cov visualization failed with code ${logits_cov_returncode}"
  exit ${logits_cov_returncode}
fi

run_logits_xj_terms_visualization
logits_xj_returncode=$?
if [[ ${logits_xj_returncode} -ne 0 ]]; then
  echo "logits xj terms visualization failed with code ${logits_xj_returncode}"
  exit ${logits_xj_returncode}
fi

run_combined_figure_visualization
combined_figure_returncode=$?
if [[ ${combined_figure_returncode} -ne 0 ]]; then
  echo "combined figure creation failed with code ${combined_figure_returncode}"
  exit ${combined_figure_returncode}
fi

echo "Analysis and covariance visualizations completed successfully."
exit 0
