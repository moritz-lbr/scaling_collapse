#!/usr/bin/env bash

#!/bin/bash

################ SLURM HEADER ################

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

# Pass the job directory that contains all folders of the separate jobs 
# Example: ./vis_submit.sh /path/to/experiment
JOB_DIR=${1:? "Please provide the directory containing all experiments as well as their configs and logs that were executed with this job-id: ./submit.sh <JOB_DIR>"}

# Count configs (skip master)
first_file="$(printf '%s\n' "$JOB_DIR"/*/simulation_config.yaml | sort | head -n 1)"
task_path="$(yq -r ".training.training_data.task" "$first_file")"
task="${task_path##*/}"
N=$(( $(yq -o=json ".network.num_hidden_layers" "$first_file") + 1 ))

if [[ "$N" -eq 0 ]]; then
  echo "No job files found in $JOB_DIR."
  exit 1
fi

# Export CONFIG_DIR to the job and set the array size
sbatch --export=ALL,JOB_DIR="$JOB_DIR",N="$N",task="$task" --array=1-"$N" vis_script.sh
