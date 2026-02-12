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
N=$(ls -1 "$JOB_DIR"/*/training_log.json  2>/dev/null | wc -l | tr -d ' ')

if [[ "$N" -eq 0 ]]; then
  echo "No job files found in $JOB_DIR."
  exit 1
fi

# Export CONFIG_DIR to the job and set the array size
sbatch --export=ALL,JOB_DIR="$JOB_DIR",N="$N" --array=1-"$N" analyze_script.sh
