#!/bin/bash

################ SLURM HEADER ################
#SBATCH --job-name=mup
#SBATCH --mail-type=ALL,ARRAY_TASKS

#SBATCH --mem=128G
#SBATCH --time=12:00:00

#SBATCH --partition=inter,cluster,th-ws,cip,cip-ws     # or whichever partition you want

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

# JOB_DIR is passed in from analyze_submit.sh -------------------------------------
: "${JOB_DIR:?JOB_DIR must be provided via analyze_submit.sh}"

# --- Paths you control -------------------------------------------------------
SLURMSCRIPTDIR=$(pwd)
SOURCEDIR="${SLURMSCRIPTDIR}"              # your source tree (contains main.py)
PROGRAM="training_analysis/compute_metrics.py"                   # your Python entrypoint
SLURMOUTDIR="${JOB_DIR}/training_analysis_slurm_output"

mkdir -p "${SLURMOUTDIR}"
exec > "${SLURMOUTDIR}/task-${SLURM_ARRAY_TASK_ID}.out" 2>&1


# --- Collect configs --------------------------------------------------------
mapfile -t JOBS < <(
    ls -1 "$JOB_DIR"/*/training_log.json 2>/dev/null | 
    sort -V
    )

TOTAL=${#JOBS[@]}
IDX=$((SLURM_ARRAY_TASK_ID - 1))

if (( TOTAL == 0 )); then
  echo "No job files found in ${JOB_DIR}."
  exit 1
fi
if (( IDX < 0 || IDX >= TOTAL )); then
  echo "Task index ${SLURM_ARRAY_TASK_ID} is out of range (have ${TOTAL} jobs). Exiting gracefully."
  exit 0
fi

SELECTED_JOB="${JOBS[$IDX]}"
JOB_PATH="${SELECTED_JOB%/*}"
OUTPUTDIR="${JOB_PATH}/weight_metrics"
echo "This task will analyze job: ${SELECTED_JOB}"

SIM_CONFIG="$JOB_PATH/simulation_config.yaml"
NUM_LAYERS="$(yq -r ".network.num_hidden_layers" "$SIM_CONFIG")"

LAYERS=("all_weights")

for ((i=0;  i<NUM_LAYERS; i++)); do
  LAYERS+=("Dense_${i}")
done

# --- Simulation Info ---------------------------------------------------------
print_info() {
    echo "# Job info"
    echo "@job_name          ${SLURM_JOB_NAME}"
    echo "@job_id            ${SLURM_JOB_ID}"
    echo "@array_job_id      ${SLURM_ARRAY_JOB_ID}"
    echo "@array_task_id     ${SLURM_ARRAY_TASK_ID}"
    echo "@start_date        $(date)"
    echo "@host              ${SLURMD_NODENAME}"
    echo "@selected_job      ${SELECTED_JOB}"
}

# --- Run the simulation -------------------------------------------------------
run_simulation() {
  pixi run python "${SOURCEDIR}/${PROGRAM}" --log-dir "${SELECTED_JOB}" --output "${OUTPUTDIR}" --layers "${LAYERS[@]}"
}

print_info

run_simulation
simulation_returncode=$?

if [ ${simulation_returncode} == 0 ]; then
    echo "Simulation was successful"
else
    echo "Simulation exited with ${simulation_returncode}. Check error logs"
fi

exit ${simulation_returncode}
