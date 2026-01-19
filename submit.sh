#!/usr/bin/env bash

#!/bin/bash

################ SLURM HEADER ################

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

# Pass the *experiment directory* that contains the "configs/" folder
# Example: ./submit.sh /path/to/experiment
CONFIG_DIR=${1:? "Please provide the directory containing configs and logs of the experiment that should be executed: ./submit.sh <EXPERIMENT_DIR>"}

# Count configs (skip master)
N=$(ls -1 "$CONFIG_DIR/configs"/*.yaml 2>/dev/null | grep -v 'master_config.yaml' | wc -l | tr -d ' ')
if [ "$N" -eq 0 ]; then
  echo "No config files found in $CONFIG_DIR/configs (excluding master_config.yaml)."
  exit 1
fi

# Export CONFIG_DIR to the job and set the array size
sbatch --export=ALL,CONFIG_DIR="$CONFIG_DIR" --array=1-"$N" slurm_script.sh
