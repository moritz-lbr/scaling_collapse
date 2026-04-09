#!/usr/bin/env bash

#!/bin/bash

################ SLURM HEADER ################

#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

################ BASH SCRIPT #################

# Pass one or more job directories that contain all folders of the separate jobs.
# Example: ./vis_submit.sh /path/to/job-1 /path/to/job-2 --compute
JOB_DIRS=()
COMPUTE_FLAG="--no-compute"
AVG_PROGRAM="training_analysis/avg_runs.py"

while (($# > 0)); do
  case "$1" in
    --compute|--no-compute)
      COMPUTE_FLAG="$1"
      shift
      ;;
    -*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      JOB_DIRS+=("$1")
      shift
      ;;
  esac
done

if [[ "${#JOB_DIRS[@]}" -eq 0 ]]; then
  echo "Please provide at least one directory containing experiments: ./vis_submit.sh <JOB_DIR> [<JOB_DIR> ...] [--compute]"
  exit 1
fi

if [[ "${#JOB_DIRS[@]}" -eq 1 ]]; then
  TARGET_JOB_DIR="${JOB_DIRS[0]}"
else
  echo "Creating averaged job directory from ${#JOB_DIRS[@]} input jobs."
  avg_output="$(pixi run python "${AVG_PROGRAM}" --log-dir "${JOB_DIRS[@]}" --layer all_weights "${COMPUTE_FLAG}")"
  echo "${avg_output}"
  TARGET_JOB_DIR="$(printf '%s\n' "${avg_output}" | tail -n 1 | sed 's/^Saved averaged runs to //')"
  if [[ -z "${TARGET_JOB_DIR}" || ! -d "${TARGET_JOB_DIR}" ]]; then
    echo "Failed to determine averaged job directory from avg_runs output."
    exit 1
  fi
fi

# Count configs (skip master)
first_file="$(printf '%s\n' "$TARGET_JOB_DIR"/*/simulation_config.yaml | sort | head -n 1)"
task_path="$(yq -r ".training.training_data.task" "$first_file")"
task="${task_path##*/}"
N=$(( $(yq -o=json ".network.num_hidden_layers" "$first_file") + 1 ))

if [[ "$N" -eq 0 ]]; then
  echo "No job files found in $TARGET_JOB_DIR."
  exit 1
fi

JOB_GROUP_NAME="${TARGET_JOB_DIR##*/}"

# Export the shared metadata to the job and pass the single target job directory as a positional argument.
sbatch --export=ALL,COMPUTE_FLAG="$COMPUTE_FLAG",N="$N",task="$task",JOB_GROUP_NAME="$JOB_GROUP_NAME" --array=1-"$N" vis_script.sh "${TARGET_JOB_DIR}"
