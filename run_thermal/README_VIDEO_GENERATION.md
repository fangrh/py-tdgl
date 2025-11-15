# Automatic Video Generation System

This system automatically generates videos from all `.h5` files in the current directory using SLURM array jobs.

## Files

1. **generate_video_array.py** - Python script that generates video from a single .h5 file
2. **generate_video.sh** - SLURM job script template
3. **submit_video_jobs.sh** - Helper script to submit jobs automatically

## How to Use

### On the SLURM Cluster

1. Navigate to the directory containing your `.h5` files:
   ```bash
   cd /path/to/your/h5files
   ```

2. Make sure the scripts are executable:
   ```bash
   chmod +x submit_video_jobs.sh generate_video.sh
   ```

3. Submit all video generation jobs at once:
   ```bash
   ./submit_video_jobs.sh
   ```

   This will:
   - Automatically detect all `.h5` files in the current directory
   - Create a SLURM array job with one task per `.h5` file
   - Each task uses 32 CPUs and 64GB RAM
   - Each task generates one video in parallel

4. Monitor job progress:
   ```bash
   squeue -u $USER
   ```

5. Check output logs:
   ```bash
   tail -f generate_video_*.out
   ```

## Output

For each `.h5` file, the system will generate:
- A corresponding `.mp4` video file (e.g., `file.h5` → `file_video.mp4`)
- Log files: `generate_video_<JobID>_<ArrayTaskID>.out` and `.err`

## Configuration

### Video Parameters (in generate_video_array.py)

```python
TIME_START = 0          # Start time
TIME_END = 8000         # End time
TIME_STEP = 50          # Time step between frames
FPS = 10                # Frames per second in output video
DPI = 100               # Image resolution
```

### Plot Options (in generate_video_array.py)

```python
PLOT_ORDER = True           # Plot |ψ|
PLOT_EPSILON = False        # Plot disorder ε
PLOT_TEMPERATURE = True     # Plot temperature
PLOT_PHASE = True           # Plot phase
PLOT_DPSI_DT = False        # Plot |dψ/dt|
```

### SLURM Resources (in generate_video.sh)

```bash
#SBATCH --cpus-per-task=32  # Number of CPU cores
#SBATCH --mem=64G           # Memory allocation
#SBATCH --time=10:00:00     # Maximum time limit
```

### Conda Environment

The script uses `conda activate tdgl` to activate the conda environment, matching the setup used in `run.sh`. Make sure conda is properly initialized in your shell environment (typically via `.bashrc` or system configuration).

## Example

If you have 9 `.h5` files in your directory:
```
hole_gap_0.0.h5
hole_gap_0.1.h5
hole_gap_0.2.h5
...
hole_gap_2.0.h5
```

Running `./submit_video_jobs.sh` will:
1. Detect all 9 files
2. Submit a job array with indices 0-8
3. Generate 9 videos in parallel:
   - `hole_gap_0.0_video.mp4`
   - `hole_gap_0.1_video.mp4`
   - etc.

## Troubleshooting

### No .h5 files found
Make sure you're in the correct directory with `.h5` files.

### Out of memory errors
Reduce `NUM_WORKERS` or increase `--mem` in `generate_video.sh`.

### FFmpeg not found
Load the appropriate module or install ffmpeg:
```bash
module load ffmpeg  # On most clusters
```

### Disk quota exceeded
- Reduce `TIME_STEP` to generate fewer frames
- Set `PLOT_DPSI_DT = False` to disable expensive calculations
- Reduce `DPI` for smaller file sizes

## Manual Submission

If you prefer to submit jobs manually:

```bash
# Count .h5 files
ls -1 *.h5 | wc -l

# Submit with specific array range (e.g., 0-8 for 9 files)
sbatch --array=0-8 generate_video.sh
```

## Notes

- Each video generation is independent and can run in parallel
- Temporary frame directories are automatically cleaned up after video creation
- The system automatically handles the `noise_strength` compatibility issue
- Videos are saved in the same directory as the `.h5` files
