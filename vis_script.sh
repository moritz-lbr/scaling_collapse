#!/bin/bash

################ SLURM HEADER ################
#SBATCH --job-name=plot_training_metrics
#SBATCH --mail-type=ALL,ARRAY_TASKS

#SBATCH --mem=128G
#SBATCH --time=12:00:00

#SBATCH --partition=inter,cluster,th-ws,cip,cip-ws   

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

# --- Paths you control -------------------------------------------------------
SLURMSCRIPTDIR=$(pwd)
SOURCEDIR="${SLURMSCRIPTDIR}"  
PROGRAM="visualizations/plot_weight_update_similarity.py"         

# A single target job directory is passed in from vis_submit.sh as a positional argument.
: "${COMPUTE_FLAG:?COMPUTE_FLAG must be provided via vis_submit.sh}"
: "${JOB_GROUP_NAME:?JOB_GROUP_NAME must be provided via vis_submit.sh}"

if [[ "$#" -ne 1 ]]; then
  echo "Exactly one target job directory must be passed to vis_script.sh."
  exit 1
fi
JOB_DIR="$1"


# --- Per-task output/error dirs ----------------------------------------------
OUTPUTDIR="${SLURMSCRIPTDIR}/figures_weight_update_similarity/${task}/${JOB_GROUP_NAME}"
SLURMOUTDIR="${OUTPUTDIR}/slurm_output"

mkdir -p "${SLURMOUTDIR}"
exec > "${SLURMOUTDIR}/task-${SLURM_ARRAY_TASK_ID}.out" 2>&1


# --- Set Layer Index ---------------------------------------------------------
IDX=$((SLURM_ARRAY_TASK_ID))

if (( IDX != $N )); then
  LAYER="Dense_$((IDX - 1))"
else 
  LAYER="all_weights"
fi

echo "This task will run for the weights of layer: ${LAYER}"


# --- Simulation Info ---------------------------------------------------------
print_info() {
    echo "# Job info"
    echo "@job_name          ${SLURM_JOB_NAME}"
    echo "@job_id            ${SLURM_JOB_ID}"
    echo "@array_job_id      ${SLURM_ARRAY_JOB_ID}"
    echo "@array_task_id     ${SLURM_ARRAY_TASK_ID}"
    echo "@start_date        $(date)"
    echo "@host              ${SLURMD_NODENAME}"
}

# --- Run the simulation -------------------------------------------------------
run_simulation() {
  pixi run python "${SOURCEDIR}/${PROGRAM}" --log-dir "${JOB_DIR}" --layer "${LAYER}" "${COMPUTE_FLAG}"
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
