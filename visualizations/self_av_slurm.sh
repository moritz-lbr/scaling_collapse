#!/usr/bin/env bash

################ SLURM HEADER ################
#SBATCH --job-name=self_av
#SBATCH --mail-type=ALL,ARRAY_TASKS
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=inter,cluster,th-ws,cip,cip-ws
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  visualizations/self_av_slurm.sh <JOB_DIR> [TASK_NAME]
  visualizations/self_av_slurm.sh <JOB_DIR> <JOB_DIR> [...] [--task-name TASK_NAME]

Options:
  --observable NAME    update_norm, test_loss, or all. Default: all
  --window-width N     Local single-job moment window width. Default: 10
  --window-stride N    Local single-job moment window stride. Default: 10
  --ylim-quantile-low Q
                      Lower y-limit quantile for plots. Default: 0.01
  --ylim-quantile-high Q
                      Upper y-limit quantile for plots. Default: 0.99
  --residual-random-seed N
                      Seed for residual diagnostic run sampling. Default: 0
  --task-name NAME     Override figures_self_av/<task-name>/.

Environment overrides:
  SELF_AV_OBSERVABLE=all
  SELF_AV_WINDOW_WIDTH=10
  SELF_AV_WINDOW_STRIDE=10
  SELF_AV_YLIM_QUANTILE_LOW=0.01
  SELF_AV_YLIM_QUANTILE_HIGH=0.99
  SELF_AV_RESIDUAL_RANDOM_SEED=0
  SELF_AV_TASK_NAME=
  SELF_AV_RUN_COMBINED=1
  SELF_AV_SKIP_STEP_PLOTS=0
  SELF_AV_FORCE_METRICS=0

Single-job mode submits one run-array for update norms and one run-array for
test losses by default, then a dependent combined plotting job.
Multi-job mode computes moments across matching independent runs at equal time
steps in the combined plotting job.
EOF
}

resolve_source_dir() {
  if [[ -n "${SOURCE_DIR:-}" ]]; then
    cd "${SOURCE_DIR}" && pwd
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/visualizations/self_averaging_metrics.py" ]]; then
    cd "${SLURM_SUBMIT_DIR}" && pwd
  elif [[ "${JOB_DIR:-}" == *"/experiments/"* && -f "${JOB_DIR%%/experiments/*}/visualizations/self_averaging_metrics.py" ]]; then
    cd "${JOB_DIR%%/experiments/*}" && pwd
  else
    cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
  fi
}

parse_job_dirs_env() {
  JOB_DIRS=()
  if [[ -n "${JOB_DIRS_JOINED:-}" ]]; then
    mapfile -t JOB_DIRS <<< "${JOB_DIRS_JOINED}"
  elif [[ -n "${JOB_DIR:-}" ]]; then
    JOB_DIRS=("${JOB_DIR}")
  fi
  if (( ${#JOB_DIRS[@]} == 0 )); then
    echo "No job directories were provided."
    exit 1
  fi
  JOB_DIR="${JOB_DIRS[0]}"
}

collect_runs() {
  mapfile -t JOBS < <(
    find "${JOB_DIR}" -mindepth 2 -maxdepth 2 -name training_log.json 2>/dev/null |
    sort -V
  )
}

configure_python() {
  export MPLCONFIGDIR="${SOURCEDIR}/.cache/matplotlib"
  mkdir -p "${MPLCONFIGDIR}"

  PYTHON_BIN=""
  if [[ -x "${SOURCEDIR}/.pixi/envs/default/bin/python" ]]; then
    PYTHON_BIN="${SOURCEDIR}/.pixi/envs/default/bin/python"
  elif [[ -x "${SOURCEDIR}/.pixi/envs/gpu/bin/python" ]]; then
    PYTHON_BIN="${SOURCEDIR}/.pixi/envs/gpu/bin/python"
  fi
}

run_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    "${PYTHON_BIN}" "$@"
  else
    pixi run python "$@"
  fi
}

validate_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "${name} must be a positive integer, got: ${value}"
    exit 1
  fi
}

validate_observable() {
  case "${1}" in
    update_norm|test_loss|all)
      ;;
    *)
      echo "observable must be update_norm, test_loss, or all; got: ${1}"
      exit 1
      ;;
  esac
}

worker_observables() {
  case "${OBSERVABLE}" in
    all)
      printf '%s\n' update_norm test_loss
      ;;
    *)
      printf '%s\n' "${OBSERVABLE}"
      ;;
  esac
}

combined_name() {
  local count="${#JOB_DIRS[@]}"
  local names=()
  local job_dir
  for job_dir in "${JOB_DIRS[@]}"; do
    names+=("$(basename "${job_dir}")")
  done
  if (( count == 1 )); then
    printf '%s\n' "${names[0]}"
  elif (( count <= 3 )); then
    local joined="${names[0]}"
    local i
    for ((i=1; i<count; i++)); do
      joined+="_${names[$i]}"
    done
    printf '%s\n' "${joined}"
  else
    printf '%s_to_%s_%sjobs\n' "${names[0]}" "${names[$((count - 1))]}" "${count}"
  fi
}

submit_mode() {
  OBSERVABLE=${SELF_AV_OBSERVABLE:-all}
  WINDOW_WIDTH=${SELF_AV_WINDOW_WIDTH:-10}
  WINDOW_STRIDE=${SELF_AV_WINDOW_STRIDE:-10}
  YLIM_QUANTILE_LOW=${SELF_AV_YLIM_QUANTILE_LOW:-0.01}
  YLIM_QUANTILE_HIGH=${SELF_AV_YLIM_QUANTILE_HIGH:-0.99}
  RESIDUAL_RANDOM_SEED=${SELF_AV_RESIDUAL_RANDOM_SEED:-0}
  TASK_NAME=${SELF_AV_TASK_NAME:-}
  JOB_DIRS=()

  while (( $# > 0 )); do
    case "${1}" in
      -h|--help)
        usage
        exit 0
        ;;
      --observable)
        OBSERVABLE="${2:?--observable requires a value}"
        shift 2
        ;;
      --window-width|--window_width)
        WINDOW_WIDTH="${2:?--window-width requires a value}"
        shift 2
        ;;
      --window-stride|--window_stride)
        WINDOW_STRIDE="${2:?--window-stride requires a value}"
        shift 2
        ;;
      --ylim-quantile-low)
        YLIM_QUANTILE_LOW="${2:?--ylim-quantile-low requires a value}"
        shift 2
        ;;
      --ylim-quantile-high)
        YLIM_QUANTILE_HIGH="${2:?--ylim-quantile-high requires a value}"
        shift 2
        ;;
      --residual-random-seed)
        RESIDUAL_RANDOM_SEED="${2:?--residual-random-seed requires a value}"
        shift 2
        ;;
      --task-name)
        TASK_NAME="${2:?--task-name requires a value}"
        shift 2
        ;;
      *)
        if [[ -d "${1}" ]]; then
          JOB_DIRS+=("$(cd "${1}" && pwd)")
        elif (( ${#JOB_DIRS[@]} == 1 )) && [[ -z "${TASK_NAME}" ]]; then
          TASK_NAME="${1}"
        else
          echo "Unrecognized argument or non-directory path: ${1}"
          usage
          exit 1
        fi
        shift
        ;;
    esac
  done

  if (( ${#JOB_DIRS[@]} == 0 )); then
    usage
    exit 1
  fi
  validate_observable "${OBSERVABLE}"
  validate_positive_int "window_width" "${WINDOW_WIDTH}"
  if (( WINDOW_WIDTH < 2 )); then
    echo "window_width must be at least 2, got: ${WINDOW_WIDTH}"
    exit 1
  fi
  validate_positive_int "window_stride" "${WINDOW_STRIDE}"
  if ! [[ "${RESIDUAL_RANDOM_SEED}" =~ ^[0-9]+$ ]]; then
    echo "residual_random_seed must be a non-negative integer, got: ${RESIDUAL_RANDOM_SEED}"
    exit 1
  fi

  JOB_DIR="${JOB_DIRS[0]}"
  SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
  SOURCEDIR="$(resolve_source_dir)"
  JOB_DIRS_JOINED="$(printf '%s\n' "${JOB_DIRS[@]}")"
  export JOB_DIR JOB_DIRS_JOINED OBSERVABLE TASK_NAME WINDOW_WIDTH WINDOW_STRIDE YLIM_QUANTILE_LOW YLIM_QUANTILE_HIGH RESIDUAL_RANDOM_SEED SOURCE_DIR="${SOURCEDIR}"

  if (( ${#JOB_DIRS[@]} == 1 )); then
    collect_runs
    N=${#JOBS[@]}
    if (( N == 0 )); then
      echo "No training logs found in ${JOB_DIR}."
      exit 1
    fi

    ARRAY_JOB_IDS=()
    while IFS= read -r worker_observable; do
      ARRAY_JOB_ID="$(
        sbatch --parsable \
          --array=1-"${N}" \
          --export=ALL \
          "${SCRIPT_PATH}" worker "${worker_observable}"
      )"
      ARRAY_JOB_IDS+=("${ARRAY_JOB_ID}")
      echo "Submitted self_av ${worker_observable} array job ${ARRAY_JOB_ID} for ${N} runs."
    done < <(worker_observables)

    if [[ "${SELF_AV_RUN_COMBINED:-1}" == "1" ]]; then
      DEPENDENCY="$(IFS=:; printf '%s' "${ARRAY_JOB_IDS[*]}")"
      COMBINE_JOB_ID="$(
        sbatch --parsable \
          --dependency=afterok:"${DEPENDENCY}" \
          --export=ALL \
          "${SCRIPT_PATH}" combine
      )"
      echo "Submitted dependent self_av plotting job ${COMBINE_JOB_ID}."
    fi
  else
    COMBINE_JOB_ID="$(
      sbatch --parsable \
        --export=ALL \
        "${SCRIPT_PATH}" combine
    )"
    echo "Submitted multi-job self_av plotting job ${COMBINE_JOB_ID} for ${#JOB_DIRS[@]} jobs."
  fi
}

worker_mode() {
  parse_job_dirs_env
  : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be provided for worker mode.}"

  WORKER_OBSERVABLE=${2:-${OBSERVABLE:-update_norm}}
  validate_observable "${WORKER_OBSERVABLE}"
  if [[ "${WORKER_OBSERVABLE}" == "all" ]]; then
    echo "Worker mode requires a concrete observable, got all."
    exit 1
  fi

  SOURCEDIR="$(resolve_source_dir)"
  SLURMOUTDIR="${JOB_DIR}/slurm_output_self_av"
  mkdir -p "${SLURMOUTDIR}"
  exec > "${SLURMOUTDIR}/task-${WORKER_OBSERVABLE}-${SLURM_ARRAY_TASK_ID}.out" 2>&1

  collect_runs
  TOTAL=${#JOBS[@]}
  IDX=$((SLURM_ARRAY_TASK_ID - 1))
  if (( IDX < 0 || IDX >= TOTAL )); then
    echo "Task index ${SLURM_ARRAY_TASK_ID} is out of range for ${TOTAL} runs."
    exit 0
  fi

  SELECTED_LOG="${JOBS[$IDX]}"
  RUN_DIR="${SELECTED_LOG%/*}"
  WINDOW_WIDTH=${WINDOW_WIDTH:-10}
  WINDOW_STRIDE=${WINDOW_STRIDE:-10}
  configure_python
  cd "${SOURCEDIR}"

  echo "# self_av worker"
  echo "@start_date        $(date)"
  echo "@host              ${SLURMD_NODENAME:-unknown}"
  echo "@array_task_id     ${SLURM_ARRAY_TASK_ID}"
  echo "@run_dir           ${RUN_DIR}"
  echo "@observable        ${WORKER_OBSERVABLE}"
  echo "@window_width      ${WINDOW_WIDTH}"
  echo "@window_stride     ${WINDOW_STRIDE}"

  run_python \
    visualizations/self_averaging_metrics.py \
    --run-dir "${RUN_DIR}" \
    --observable "${WORKER_OBSERVABLE}" \
    --window-width "${WINDOW_WIDTH}" \
    --window-stride "${WINDOW_STRIDE}" \
    --no-progress

  echo "self_av ${WORKER_OBSERVABLE} analysis completed for ${RUN_DIR}."
}

combine_mode() {
  parse_job_dirs_env

  SOURCEDIR="$(resolve_source_dir)"
  SLURMOUTDIR="${JOB_DIR}/slurm_output_self_av"
  mkdir -p "${SLURMOUTDIR}"
  if (( ${#JOB_DIRS[@]} == 1 )); then
    exec > "${SLURMOUTDIR}/combined.out" 2>&1
  else
    exec > "${SLURMOUTDIR}/combined-$(combined_name).out" 2>&1
  fi

  OBSERVABLE=${OBSERVABLE:-all}
  WINDOW_WIDTH=${WINDOW_WIDTH:-10}
  WINDOW_STRIDE=${WINDOW_STRIDE:-10}
  YLIM_QUANTILE_LOW=${YLIM_QUANTILE_LOW:-0.01}
  YLIM_QUANTILE_HIGH=${YLIM_QUANTILE_HIGH:-0.99}
  RESIDUAL_RANDOM_SEED=${RESIDUAL_RANDOM_SEED:-0}
  validate_observable "${OBSERVABLE}"
  if ! [[ "${RESIDUAL_RANDOM_SEED}" =~ ^[0-9]+$ ]]; then
    echo "residual_random_seed must be a non-negative integer, got: ${RESIDUAL_RANDOM_SEED}"
    exit 1
  fi
  configure_python
  cd "${SOURCEDIR}"

  echo "# self_av combined plots"
  echo "@start_date        $(date)"
  echo "@host              ${SLURMD_NODENAME:-unknown}"
  echo "@job_dirs          ${JOB_DIRS[*]}"
  echo "@observable        ${OBSERVABLE}"
  echo "@window_width      ${WINDOW_WIDTH}"
  echo "@window_stride     ${WINDOW_STRIDE}"
  echo "@ylim_quantile_low ${YLIM_QUANTILE_LOW}"
  echo "@ylim_quantile_high ${YLIM_QUANTILE_HIGH}"
  echo "@residual_random_seed ${RESIDUAL_RANDOM_SEED}"

  ARGS=(visualizations/plot_self_averaging.py --observable "${OBSERVABLE}")
  local job_dir
  for job_dir in "${JOB_DIRS[@]}"; do
    ARGS+=(--job-dir "${job_dir}")
  done
  ARGS+=(--window-width "${WINDOW_WIDTH}" --window-stride "${WINDOW_STRIDE}")
  ARGS+=(--ylim-quantile-low "${YLIM_QUANTILE_LOW}" --ylim-quantile-high "${YLIM_QUANTILE_HIGH}")
  ARGS+=(--residual-random-seed "${RESIDUAL_RANDOM_SEED}")
  if (( ${#JOB_DIRS[@]} == 1 )); then
    ARGS+=(--compute-missing)
  fi
  if [[ "${SELF_AV_FORCE_METRICS:-0}" == "1" ]]; then
    ARGS+=(--force-metrics)
  fi
  if [[ -n "${TASK_NAME:-}" ]]; then
    ARGS+=(--task-name "${TASK_NAME}")
  fi
  if [[ "${SELF_AV_SKIP_STEP_PLOTS:-0}" == "1" ]]; then
    ARGS+=(--skip-step-plots)
  fi
  run_python "${ARGS[@]}"

  echo "self_av combined plots completed."
}

MODE=${1:-submit}
case "${MODE}" in
  worker)
    worker_mode "$@"
    ;;
  combine)
    combine_mode
    ;;
  *)
    submit_mode "$@"
    ;;
esac
