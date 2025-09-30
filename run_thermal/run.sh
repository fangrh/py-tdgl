#!/bin/bash -l
#SBATCH --no-requeue
#SBATCH --get-user-env
#SBATCH --job-name=tdgl_gamma_array
#SBATCH --output=tdgl_gamma_%A_%a.out
#SBATCH --error=tdgl_gamma_%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-12

# Calculate gamma value based on array index
# Array indices: 0,1,2,3,4,5,6,7,8,9,10
# Gamma values: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9
T_heat_values=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
T_heat=${T_heat_values[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with gamma = $gamma"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Activate the conda environment
conda activate tdgl
# Get start time
start_time=$(date +%s)

# Run the Python script with the calculated gamma value
python run_tdgl_with_different_gamma.py --T_heat $T_heat --hole_gap 2.5

# Get end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))

# Convert seconds to hours, minutes, seconds
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"

echo "Completed job $SLURM_ARRAY_TASK_ID with gamma = $gamma" 