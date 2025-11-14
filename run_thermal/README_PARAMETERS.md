# run_tdgl_with_different_gamma.py Parameter Documentation

## Command Line Parameters

### Physical Parameters
- `--gamma`: Gamma parameter (default: 1.0)
- `--u`: u parameter (default: 5.79)
- `--d`: Film thickness (default: 0.01)
- `--xi`: Coherence length (default: 0.25)
- `--london_lambda`: London penetration depth (default: 0.25)
- `--sigma`: Normal state conductivity (default: 1000)

### Thermal Diffusion Parameters
- `--use_heat`: Whether to enable thermal diffusion (default: True)
- `--T_0`: Dimensionless critical temperature (default: 1.0)
- `--kappa_eff`: Effective thermal conductivity (dimensionless) (default: 0.02)
- `--C_eff`: Effective heat capacity (dimensionless) (default: 1.0)
- `--T_heat`: Environment temperature (dimensionless) (default: 0.1)
- `--hole_eta`: Heat exchange coefficient for hole region (default: 0.05)
- `--environment_eta`: Heat exchange coefficient for external region (default: 10.0)
- `--hole_gap`: Half-width of hole region (μm) (default: 1.0)

### Geometry Parameters
- `--width`: Film width (μm) (default: 6.0)
- `--length`: Film length (μm) (default: 18.0)
- `--electride_distance`: Distance between electrodes (μm) (default: 8.0)
- `--probe_distance`: Distance of probe point from center (μm) (default: 2.5)

### Time Parameters ⭐ New Addition
- `--ramp_up_time`: Current ramp-up time (default: 5000)
- `--max_current_time`: Duration to hold at maximum current (default: 0)
- `--ramp_down_time`: Current ramp-down time (default: 5000)
- `--zero_current_time`: Duration to hold at zero current (default: 0)

**Total simulation time**: `solve_time = ramp_up_time + max_current_time + ramp_down_time + zero_current_time`

## Usage Examples

### Example 1: Basic run (using default parameters)
```bash
python run_tdgl_with_different_gamma.py
```

### Example 2: Modify gamma and thermal parameters
```bash
python run_tdgl_with_different_gamma.py \
    --gamma 2.0 \
    --hole_eta 0.1 \
    --environment_eta 20.0 \
    --kappa_eff 0.05
```

### Example 3: Modify time parameters (quick test)
```bash
python run_tdgl_with_different_gamma.py \
    --ramp_up_time 1000 \
    --ramp_down_time 1000
# Total time = 1000 + 0 + 1000 + 0 = 2000
```

### Example 4: Longer simulation time
```bash
python run_tdgl_with_different_gamma.py \
    --ramp_up_time 10000 \
    --max_current_time 5000 \
    --ramp_down_time 10000 \
    --zero_current_time 2000
# Total time = 10000 + 5000 + 10000 + 2000 = 27000
```

### Example 5: Complete parameter combination
```bash
python run_tdgl_with_different_gamma.py \
    --gamma 1.5 \
    --hole_eta 0.05 \
    --environment_eta 15.0 \
    --kappa_eff 0.03 \
    --T_heat 0.15 \
    --hole_gap 2.4 \
    --width 8.0 \
    --length 20.0 \
    --ramp_up_time 6000 \
    --max_current_time 1000 \
    --ramp_down_time 6000
```

## Current Profile Description

The current varies over time in four stages:

```
Current (I)
   │
Imax│    ┌──────────┐ Stage 2: max_current_time
   │   ╱│          │╲
   │  ╱ │          │ ╲
   │ ╱  │          │  ╲
  0├────┼──────────┼───┼─────── Time (t)
      Stage 1    Stage 3  Stage 4
   ramp_up_time  ramp_down zero_current
                   _time      _time
```

**Stage 1**: Linear increase from 0 to Imax (duration: `ramp_up_time`)
**Stage 2**: Hold at Imax (duration: `max_current_time`)
**Stage 3**: Linear decrease from Imax to 0 (duration: `ramp_down_time`)
**Stage 4**: Hold at 0 (duration: `zero_current_time`)

## Output File Naming

Output filenames include all important parameters:
```
gamma{gamma}_holeeta{hole_eta}_environmenteta{environment_eta}_...npz
```

Saved data includes:
- `time`: Time array
- `currents`: Current array
- `voltages`: Voltage difference array
- All input parameters (including time parameters)

## Important Notes

1. **Time units**: All time parameters are dimensionless (normalized time)
2. **Thermal diffusion stability**:
   - Too small `ramp_up_time` may lead to numerical instability
   - Recommend `ramp_up_time >= 1000` to ensure stability
3. **Simulation duration**:
   - Total time too long will increase computational cost
   - Recommend testing with shorter time first, then extend after verification
4. **hole_gap units**: Note that `hole_gap` is in **μm** (physical units), and will be automatically converted to dimensionless units in the code

## Performance Optimization Suggestions

- **Quick test**: `--ramp_up_time 500 --ramp_down_time 500`
- **Standard simulation**: `--ramp_up_time 5000 --ramp_down_time 5000` (default)
- **Detailed study**: `--ramp_up_time 10000 --max_current_time 2000 --ramp_down_time 10000`

## Batch Processing Script Example

```bash
#!/bin/bash
# Scan different gamma values and time parameters

for gamma in 0.5 1.0 1.5 2.0; do
    for ramp_time in 3000 5000 7000; do
        python run_tdgl_with_different_gamma.py \
            --gamma $gamma \
            --ramp_up_time $ramp_time \
            --ramp_down_time $ramp_time \
            --hole_eta 0.05 \
            --environment_eta 10.0
    done
done
```
