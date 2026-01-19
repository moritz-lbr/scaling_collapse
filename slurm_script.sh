#!/bin/bash

################ SLURM HEADER ################
#SBATCH --job-name=mup
#SBATCH --mail-type=ALL,ARRAY_TASKS

#SBATCH --mem=16G
#SBATCH --time=12:00:00

#SBATCH --partition=cluster,th-ws,cip,cip-ws     # or whichever partition you want
#SBATCH --gres=gpu:1           # at least one GPU

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

# --- Paths you control -------------------------------------------------------
SLURMSCRIPTDIR=$(pwd)
SOURCEDIR="${SLURMSCRIPTDIR}/src"              # your source tree (contains main.py)
PROGRAM="run_experiment.py"                   # your Python entrypoint

# CONFIG_DIR is passed in from submit.sh and must contain a "configs/" folder.
: "${CONFIG_DIR:?CONFIG_DIR must be provided via submit.sh}"

CONFIG_ROOT="${SLURMSCRIPTDIR}/${CONFIG_DIR}/configs"


# --- Per-task output/error dirs ----------------------------------------------
JOBFOLDER="job-${SLURM_ARRAY_JOB_ID}/task-${SLURM_ARRAY_TASK_ID}"
WORKDIR="/scratch-local/${USER}/${JOBFOLDER}"
OUTPUTDIR="${SLURMSCRIPTDIR}/${CONFIG_DIR}/logs/job-${SLURM_ARRAY_JOB_ID}"
SLURMOUTDIR="${OUTPUTDIR}/slurm_output"

mkdir -p "${SLURMOUTDIR}"
exec > "${SLURMOUTDIR}/task-${SLURM_ARRAY_TASK_ID}.out" 2>&1


# --- Collect configs --------------------------------------------------------
mapfile -t CONFIGS < <(
    ls -1 "${CONFIG_ROOT}/"*.yaml 2>/dev/null | 
    grep -v 'master_config.yaml' | 
    sort -V
    )

TOTAL=${#CONFIGS[@]}
IDX=$((SLURM_ARRAY_TASK_ID - 1))

if (( TOTAL == 0 )); then
  echo "No config files found in ${CONFIG_ROOT}."
  exit 1
fi
if (( IDX < 0 || IDX >= TOTAL )); then
  echo "Task index ${SLURM_ARRAY_TASK_ID} is out of range (have ${TOTAL} configs). Exiting gracefully."
  exit 0
fi

SELECTED_CONFIG="${CONFIGS[$IDX]}"
echo "This task will run config: ${SELECTED_CONFIG}"


# --- Simulation Info ---------------------------------------------------------
print_info() {
    echo "# Job info"
    echo "@job_name          ${SLURM_JOB_NAME}"
    echo "@job_id            ${SLURM_JOB_ID}"
    echo "@array_job_id      ${SLURM_ARRAY_JOB_ID}"
    echo "@array_task_id     ${SLURM_ARRAY_TASK_ID}"
    echo "@start_date        $(date)"
    echo "@host              ${SLURMD_NODENAME}"
    echo "@working_directory ${WORKDIR}"
}

# --- Run the simulation -------------------------------------------------------
run_simulation() {
  pixi run -e gpu python "${SOURCEDIR}/${PROGRAM}" --path "${SELECTED_CONFIG}" --output_dir "${OUTPUTDIR}"
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
