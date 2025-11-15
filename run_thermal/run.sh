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
#SBATCH --array=0-8

# ============================================
# CONFIGURATION: Choose tdgl version
# ============================================
# Set USE_LOCAL_TDGL=1 to use local development version
# Set USE_LOCAL_TDGL=0 to use conda environment version (default)
USE_LOCAL_TDGL=1

# Calculate gamma value based on array index
# Array indices: 0,1,2,3,4,5,6,7,8,9,10
# Gamma values: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9
hole_gap_values=(0.0 0.1 0.2 0.25 0.5 0.8 1.0 1.5 2.0)
hole_gap=${hole_gap_values[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with T_heat = $T_heat"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Activate the conda environment
conda activate tdgl
# Get start time
start_time=$(date +%s)

# Build command with optional --use_local_tdgl flag
python run_tdgl_with_different_gamma.py --hole_gap $hole_gap --ramp_up_time 4000 --ramp_down_time 4000 --save_hdf5