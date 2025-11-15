#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --get-user-env
#SBATCH --job-name=generate_video
#SBATCH --output=generate_video_%A_%a.out
#SBATCH --error=generate_video_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

# This script will be submitted with --array parameter set dynamically
# based on the number of .h5 files found

echo "Starting video generation job"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"

module load ffmpeg

# Activate the conda environment
conda activate tdgl

# Get start time
start_time=$(date +%s)

# Run the video generation script
# The script will automatically select the .h5 file based on SLURM_ARRAY_TASK_ID
python generate_video_array.py

# Get end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Job completed in $duration seconds"
