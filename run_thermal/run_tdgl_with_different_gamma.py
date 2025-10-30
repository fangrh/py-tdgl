#%% Imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import tdgl
from tdgl.geometry import box
from tdgl.visualization.animate import create_animation
from IPython.display import HTML, display
import h5py

# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--hole_eta', type=float, default=0.05)
parser.add_argument('--environment_eta', type=float, default=10.0)
parser.add_argument('--C_eff', type=float, default=1.0)
parser.add_argument('--T_heat', type=float, default=0.1)
parser.add_argument("--T_0", type=float, default=1.0)
parser.add_argument('--kappa_eff', type=float, default=0.02)
parser.add_argument("--hole_gap", type=float, default=1.0)
parser.add_argument("--d", type=float, default=0.01)
parser.add_argument("--xi", type=float, default=0.25)
parser.add_argument("--london_lambda", type=float, default=0.25)
parser.add_argument("--sigma", type=float, default=1e3)
parser.add_argument("--use_heat", type=bool, default=True)
parser.add_argument("--u", type=float, default=5.79)
parser.add_argument("--width", type=float, default=6.0)
parser.add_argument("--electride_distance", type=float, default=8.0)
parser.add_argument("--probe_distance", type=float, default=2.5)
parser.add_argument("--length", type=float, default=18.0)
parser.add_argument("--ramp_up_time", type=float, default=5000)
parser.add_argument("--max_current_time", type=float, default=0)
parser.add_argument("--ramp_down_time", type=float, default=5000)
parser.add_argument("--zero_current_time", type=float, default=0)
args = parser.parse_args() 


length_units = "um"
xi = args.xi  # coherence length   
london_lambda = args.london_lambda  # London penetration depth
d = args.d  # thickness
gamma = args.gamma
u = args.u
sigma = args.sigma
use_heat = args.use_heat
T_0 = args.T_0            # Dimensionless critical temperature
kappa_eff = args.kappa_eff    # Effective thermal conductivity (dimensionless)
hole_eta = args.hole_eta           # Heat exchange coefficient with environment (dimensionless)
environment_eta = args.environment_eta           # Heat exchange coefficient with environment (dimensionless)
C_eff = args.C_eff        # Effective heat capacity (dimensionless)
T_heat = args.T_heat

layer = tdgl.Layer(coherence_length=xi, 
                   london_lambda=london_lambda, 
                   thickness=d, 
                   gamma=gamma, 
                   u=u,
                   conductivity=sigma,
                #    use_heat=use_heat,
                    T_0=T_0,            # Dimensionless critical temperature
                    kappa_eff=kappa_eff,    # Effective thermal conductivity (dimensionless)
                    eta=hole_eta,           # Heat exchange coefficient with environment (dimensionless)
                    C_eff=C_eff,        # Effective heat capacity (dimensionless)
                    T_heat = T_heat)


# Superconductor disorder parameters
epsilon_normal = 1.0  # Epsilon value for normal superconducting regions
epsilon_strip = 1.0  # Epsilon value for middle strip region
strip_half_width = 0.1 / 2  # Half width of middle strip

#%% Device geometry
width = args.width  # total width
length = args.length  # total length
# metal_strip_width = 0.1  # width of non-superconducting strip in μm
electride_distance = args.electride_distance  # Distance of superconductor from endpoints
extend = (length-electride_distance)/2  # Extension length at both ends

# Create internal electrodes (no longer at endpoints)
terminal_width = width+2  # Electrode width
terminal_thickness = 1  # Electrode thickness

# Calculate electrode positions so both ends extend by length 'extend'
top_terminal_y = length/2 - extend - terminal_thickness/2  # Top electrode y-coordinate
bottom_terminal_y = -top_terminal_y  # Bottom electrode y-coordinate (symmetric position)


# Create the superconducting film (single piece)
film = (
    tdgl.Polygon("film", points=box(width, length))
    .resample(401)
    .buffer(0.01)
)


# Create current terminals
source = (
    tdgl.Polygon("source", points=box(terminal_width, terminal_thickness))  # Electrode dimensions
    .translate(dy=top_terminal_y)  # Located at top, but not at endpoint
)
drain = (
    tdgl.Polygon("drain", points=box(terminal_width, terminal_thickness))
    .translate(dy=bottom_terminal_y)  # Located at bottom, but not at endpoint
)

# Add probe points for voltage measurement (between terminals)
# Fixed probe point positions at y=probe_distance and y=-probe_distance
probe_points = [(0, args.probe_distance), (0, -args.probe_distance)]  # Fixed position probe points


#%% Create device and generate mesh
device = tdgl.Device(
    f"rectangle_with_heat_lambda_{london_lambda}_xi_{xi}_u_{u}_gamma_{gamma}_sigma_{sigma}_T_0_{T_0}_kappa_eff_{kappa_eff}_C_eff_{C_eff}_T_heat_{T_heat}",
    layer=layer,
    film=film,
    terminals=[source, drain],
    probe_points=probe_points,
    length_units=length_units,
)

# Generate mesh
device.make_mesh(max_edge_length=xi/2, smooth=100)

# Convert hole_gap from physical units (um) to dimensionless units (in xi)
# device.mesh.sites are in dimensionless units (normalized by xi)
hole_gap = args.hole_gap  # Physical gap size in um
hole_gap_dimensionless = hole_gap / xi  # Convert to dimensionless units

y_coords = device.mesh.sites[:, 1]  # Dimensionless coordinates (in units of xi)
mask = (y_coords >= -hole_gap_dimensionless) & (y_coords <= hole_gap_dimensionless)
eta_arr = np.full(len(device.mesh.sites), environment_eta)
eta_arr[mask] = hole_eta
device.layer.eta = eta_arr

# Print diagnostic info
print(f"hole_gap (physical): {hole_gap} um")
print(f"hole_gap (dimensionless): {hole_gap_dimensionless} xi")
print(f"xi = {xi} um")
print(f"Number of mesh points in hole region: {np.sum(mask)} / {len(y_coords)} ({100*np.sum(mask)/len(y_coords):.1f}%)")

#%% Define disorder function
def disorder_function(r, **kwargs):
    """Creates a non-superconducting strip in the middle.
    
    Args:
        r: Position vector (x, y) or (x, y, z)
        
    Returns:
        epsilon_normal for normal superconducting regions
        epsilon_strip for middle strip region
    """
    # Extract y-coordinate from position vector
    y = r[1]  # r = [x, y] or r = [x, y, z]
    
    # Define the strip in the middle with specified epsilon
    if -strip_half_width <= y <= strip_half_width:
        return epsilon_strip  # Middle strip region
    else:
        return epsilon_normal  # Normal superconducting region
    



# Define time segments for four-stage current (from command line arguments)
ramp_up_time = args.ramp_up_time    # Stage 1: Time to ramp up to maximum
max_current_time = args.max_current_time  # Stage 2: Time to hold maximum value
ramp_down_time = args.ramp_down_time # Stage 3: Time to ramp down to zero
zero_current_time = args.zero_current_time  # Stage 4: Time to hold at zero
solve_time = ramp_up_time + max_current_time + ramp_down_time + zero_current_time
# Define four-stage time-dependent current function
def terminal_currents(t):
    """Four-stage current function:
    1. Stage 1: Current ramps linearly from 0 to maximum
    2. Stage 2: Hold at maximum value for a certain time
    3. Stage 3: Ramp down from maximum to 0
    4. Stage 4: Hold at 0
    """
    max_current = 3000 * d/0.08 * width/6 # Maximum current value

    # Calculate time segment boundaries
    t1 = ramp_up_time  # End of stage 1
    t2 = t1 + max_current_time  # End of stage 2
    t3 = t2 + ramp_down_time  # End of stage 3

    if t < t1:
        # Stage 1: Linear ramp from 0 to maximum
        current = max_current * (t / ramp_up_time)
    elif t < t2:
        # Stage 2: Hold at maximum
        current = max_current
    elif t < t3:
        # Stage 3: Linear ramp down from maximum to 0
        current = max_current * (1 - (t - t2) / ramp_down_time)
    else:
        # Stage 4: Hold at 0
        current = 0

    return dict(source=current, drain=-current)


options = tdgl.SolverOptions(
    solve_time=solve_time,
    # output_file="disorder_strip_iv_curve.h5",
    field_units="mT",
    current_units="uA",
    dt_max=1e-2,
    # save_every=50,  # Save more frequently for smoother animation
    include_screening=False,
    max_solve_retries=100,
    adaptive=True
)

# Run simulation with zero field, time-dependent current, and disorder
solution = tdgl.solve(
    device,
    options,
    applied_vector_potential=0,  # Zero applied field
    terminal_currents=terminal_currents,  # Pass the function directly
    disorder_epsilon=disorder_function,  # Use our disorder function
    use_heat=use_heat
)

# Extract time data
time_data = solution.dynamics.time

# Calculate current values at each time point
currents = np.array([terminal_currents(t)['source'] for t in time_data])

# Get voltage data - voltage difference between probe points
voltage_data = solution.dynamics.mu
print(f"Voltage data shape: {voltage_data.shape}")

# Handle voltage data dimensions correctly
# If voltage_data has shape (2, n), need to transpose to (n, 2)
if voltage_data.shape[0] == 2 and len(voltage_data.shape) == 2:
    voltage_data = voltage_data.T  # Transpose data to make shape (n, 2)
    print(f"Transposed voltage data shape: {voltage_data.shape}")
    voltage_diff = voltage_data[:, 0] - voltage_data[:, 1]
elif len(voltage_data.shape) == 2 and voltage_data.shape[1] == 2:
    # Data already has shape (n, 2)
    voltage_diff = voltage_data[:, 0] - voltage_data[:, 1]
else:
    # If shape is not (2, n), try to get probe voltages directly
    print("Trying to use probe voltages directly...")
    probe_voltages = solution.dynamics.mu
    if len(probe_voltages) == 2:  # If there are two sequences, each corresponding to a probe point
        voltage_diff = probe_voltages[0] - probe_voltages[1]
    else:
        # Last resort, directly use first probe point voltage
        voltage_diff = probe_voltages[0]

# solution.dynamics.mu is dimensionless, needs to be multiplied by voltage scale V0 to get actual voltage
try:
    # Try to get voltage scale V0
    V0 = device.V0()  # Voltage scale, in units of V
    print(f"Voltage scale (V0): {V0}")

    # Convert dimensionless voltage to microvolts
    voltage_scale_uV = V0.to("uV").magnitude
    voltage_diff_uV = voltage_diff * voltage_scale_uV
    voltage_unit = "μV"
except ValueError:
    # If conductivity is not defined, cannot get V0, use dimensionless values
    print("Conductivity not defined. Using dimensionless values for voltage.")
    voltage_diff_uV = voltage_diff
    voltage_unit = "a.u. (dimensionless)"

print(f"Current data shape: {currents.shape}")
print(f"Voltage difference shape: {voltage_diff.shape}")

# Ensure data lengths match
min_length = min(len(currents), len(voltage_diff_uV))
currents = currents[:min_length]
voltage_diff_uV = voltage_diff_uV[:min_length]


np.savez(f"gamma{gamma}_holeeta{hole_eta}_environmenteta{environment_eta}_Theat{T_heat}_d{d}_xi{xi}_londonlambda{london_lambda}_sigma{sigma}_u{u}_useheat{use_heat}_T0{T_0}_kappaeff{kappa_eff}_Ceff{C_eff}_width{width}_holegap{hole_gap}_electride_distance{electride_distance}_probe_distance{args.probe_distance}_length{args.length}.npz", 
         time=time_data, 
         currents=currents, 
         voltages=voltage_diff_uV, 
         gamma=gamma, 
         d=d, 
         u=u, 
         sigma=sigma, 
         xi=xi, 
         london_lambda=london_lambda, 
         use_heat=use_heat, 
         T_0=T_0, 
         kappa_eff=kappa_eff, 
         hole_eta=hole_eta, 
         environment_eta=environment_eta, 
         C_eff=C_eff, 
         T_heat=T_heat, 
         width=width, 
         hole_gap=hole_gap, 
         electride_distance=electride_distance,
         probe_distance=args.probe_distance,
         length=args.length,
         ramp_up_time=ramp_up_time, 
         max_current_time=max_current_time, 
         ramp_down_time=ramp_down_time, 
         zero_current_time=zero_current_time, 
         solve_time=solve_time)