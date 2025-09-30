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
args = parser.parse_args() 


length_units = "um"
xi = args.xi  # coherence length   
london_lambda = args.london_lambda  # London penetration depth
d = args.d  # thickness
gamma = args.gamma
u = args.u
sigma = args.sigma
use_heat = args.use_heat
T_0 = args.T_0            # 无量纲临界温度
kappa_eff = args.kappa_eff    # 有效热导率 (无量纲)
hole_eta = args.hole_eta           # 与环境的热交换系数 (无量纲)
environment_eta = args.environment_eta           # 与环境的热交换系数 (无量纲)
C_eff = args.C_eff        # 有效热容 (无量纲)
T_heat = args.T_heat

layer = tdgl.Layer(coherence_length=xi, 
                   london_lambda=london_lambda, 
                   thickness=d, 
                   gamma=gamma, 
                   u=u, 
                   conductivity=sigma,
                #    use_heat=use_heat,
                    T_0=T_0,            # 无量纲临界温度
                    kappa_eff=kappa_eff,    # 有效热导率 (无量纲)
                    eta=hole_eta,           # 与环境的热交换系数 (无量纲)
                    C_eff=C_eff,        # 有效热容 (无量纲)
                    T_heat = T_heat)


# 超导体disorder参数
epsilon_normal = 1.0  # 普通超导区域的epsilon值
epsilon_strip = 1.0  # 中间条带区域的epsilon值
strip_half_width = 0.1 / 2  # 中间条带的半宽度

#%% Device geometry
width = args.width  # total width
length = args.length  # total length
# metal_strip_width = 0.1  # width of non-superconducting strip in μm
electride_distance = args.electride_distance  # 超导体距离两端的长度
extend = (length-electride_distance)/2  # 两端露出的长度

# 创建内部电极（不再位于两端）
terminal_width = width+2  # 电极宽度
terminal_thickness = 1  # 电极厚度

# 计算电极位置，使两端各露出extend长度
top_terminal_y = length/2 - extend - terminal_thickness/2  # 上电极y坐标
bottom_terminal_y = -top_terminal_y  # 下电极y坐标（对称位置）


# Create the superconducting film (single piece)
film = (
    tdgl.Polygon("film", points=box(width, length))
    .resample(401)
    .buffer(0.01)
)


# 创建电流端子
source = (
    tdgl.Polygon("source", points=box(terminal_width, terminal_thickness))  # 电极尺寸
    .translate(dy=top_terminal_y)  # 位于上部，但不在端部
)
drain = (
    tdgl.Polygon("drain", points=box(terminal_width, terminal_thickness))
    .translate(dy=bottom_terminal_y)  # 位于下部，但不在端部
)

# Add probe points for voltage measurement (between terminals)
# 固定探测点位置在y=5.0和y=2.5
probe_points = [(0, args.probe_distance), (0, -args.probe_distance)]  # 固定位置的探测点


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

hole_gap = args.hole_gap
y_coords = device.mesh.sites[:, 1]
mask = (y_coords >= -hole_gap) & (y_coords <= hole_gap)
eta_arr = np.full(len(device.mesh.sites), environment_eta)
eta_arr[mask] = hole_eta
device.layer.eta = eta_arr

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
    
    
    

# 定义四段式电流的时间段
ramp_up_time = 5000    # 第一段：上升到最大值的时间
max_current_time = 0  # 第二段：保持最大值的时间
ramp_down_time = 5000 # 第五段降到零
zero_current_time = 0  # 第四段：保持零的时间
solve_time = ramp_up_time + max_current_time + ramp_down_time + zero_current_time
# 定义四段式时间依赖的电流函数
def terminal_currents(t):
    """四段式电流函数:
    1. 第一段：电流从0线性上升到最大值
    2. 第二段：保持最大值一定时间
    3. 第三段：从最大值下降到0
    4. 第四段：保持0
    """
    max_current = 3000 * d/0.08 * width/6 # 最大电流值
    
    # 计算时间段边界
    t1 = ramp_up_time  # 第一段结束
    t2 = t1 + max_current_time  # 第二段结束
    t3 = t2 + ramp_down_time  # 第三段结束
    
    if t < t1:
        # 第一段：从0线性上升到最大值
        current = max_current * (t / ramp_up_time)
    elif t < t2:
        # 第二段：保持最大值
        current = max_current
    elif t < t3:
        # 第三段：从最大值线性下降到0
        current = max_current * (1 - (t - t2) / ramp_down_time)
    else:
        # 第四段：保持为0
        current = 0
        
    return dict(source=current, drain=-current)


options = tdgl.SolverOptions(
    solve_time=solve_time,
    # output_file="disorder_strip_iv_curve.h5",
    field_units="mT",
    current_units="uA",
    dt_max=1e-2,
    # save_every=50,  # 更频繁地保存数据以获得更平滑的动画
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

# 提取时间数据
time_data = solution.dynamics.time

# 计算每个时间点的电流值
currents = np.array([terminal_currents(t)['source'] for t in time_data])

# 获取电压数据 - 探测点之间的电压差
voltage_data = solution.dynamics.mu
print(f"Voltage data shape: {voltage_data.shape}")

# 正确处理电压数据维度
# 如果voltage_data的形状是(2, n)，需要转置为(n, 2)
if voltage_data.shape[0] == 2 and len(voltage_data.shape) == 2:
    voltage_data = voltage_data.T  # 转置数据使形状变为(n, 2)
    print(f"Transposed voltage data shape: {voltage_data.shape}")
    voltage_diff = voltage_data[:, 0] - voltage_data[:, 1]
elif len(voltage_data.shape) == 2 and voltage_data.shape[1] == 2:
    # 数据已经是(n, 2)形状
    voltage_diff = voltage_data[:, 0] - voltage_data[:, 1]
else:
    # 如果形状是(2, n)以外的形状，尝试直接获取探测点的电压
    print("Trying to use probe voltages directly...")
    probe_voltages = solution.dynamics.mu
    if len(probe_voltages) == 2:  # 如果有两个序列，每个对应一个探测点
        voltage_diff = probe_voltages[0] - probe_voltages[1]
    else:
        # 最后的尝试，直接使用第一个探测点的电压
        voltage_diff = probe_voltages[0]
        
# solution.dynamics.mu是无量纲的，需要乘以电压标度V0得到实际电压
try:
    # 尝试获取电压标度V0
    V0 = device.V0()  # 电压标度，单位为V
    print(f"Voltage scale (V0): {V0}")
    
    # 将无量纲电压转换为微伏特
    voltage_scale_uV = V0.to("uV").magnitude
    voltage_diff_uV = voltage_diff * voltage_scale_uV
    voltage_unit = "μV"
except ValueError:
    # 如果conductivity未定义，无法获取V0，则使用无量纲数值
    print("Conductivity not defined. Using dimensionless values for voltage.")
    voltage_diff_uV = voltage_diff
    voltage_unit = "a.u. (dimensionless)"

print(f"Current data shape: {currents.shape}")
print(f"Voltage difference shape: {voltage_diff.shape}")

# 确保数据长度匹配
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