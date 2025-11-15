#!/bin/bash
# Helper script to submit video generation jobs
# Automatically detects .h5 files and submits array job

# Count .h5 files in current directory
h5_files=(*.h5)
num_files=${#h5_files[@]}

# Check if any .h5 files exist
if [ "$num_files" -eq 0 ]; then
    echo "ERROR: No .h5 files found in current directory"
    exit 1
fi

# If there's only one file with wildcard pattern, it means no .h5 files
if [ "$num_files" -eq 1 ] && [ "${h5_files[0]}" = "*.h5" ]; then
    echo "ERROR: No .h5 files found in current directory"
    exit 1
fi

echo "Found $num_files .h5 file(s):"
for i in "${!h5_files[@]}"; do
    echo "  [$i] ${h5_files[$i]}"
done
echo ""

# Calculate array range (0 to num_files-1)
array_max=$((num_files - 1))

echo "Submitting SLURM array job with array range: 0-$array_max"
echo "Each job will use 32 CPUs and 64GB RAM"
echo ""

# Submit the job
sbatch --array=0-$array_max generate_video.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "Job submitted successfully!"
    echo "Monitor job status with: squeue -u \$USER"
    echo "Check output with: tail -f generate_video_*.out"
else
    echo ""
    echo "ERROR: Job submission failed"
    exit 1
fi
