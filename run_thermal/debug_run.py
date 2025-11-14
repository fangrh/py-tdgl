#%%
"""
调试运行脚本 - 参数内部设置
基于 run_tdgl_with_different_gamma.py
包含 device 可视化和本地 tdgl
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 使用本地的 tdgl
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tdgl
from tdgl.geometry import box
import h5py

print(f"使用 tdgl 路径: {tdgl.__file__}")

#%% ========== 参数设置 ==========
print("="*60)
print("参数设置")
print("="*60)

# 材料参数
gamma = 1.0
hole_eta = 0.05
environment_eta = 10.0
C_eff = 1.0
T_heat = 0.1
T_0 = 1.0
kappa_eff = 0.02
hole_gap = 1.0  # um
d = 0.01  # um
xi = 0.25  # um
london_lambda = 0.25  # um
sigma = 1e3
use_heat = True
u = 5.79
suppress_electrode_edge_heating = True

# 几何参数
width = 6.0  # um
electride_distance = 6.5  # um
probe_distance = 2.0  # um
length = 6.0  # um

# 时间参数
ramp_up_time = 4000
max_current_time = 0
ramp_down_time = 4000
zero_current_time = 0

# 噪声参数
noise_strength = 0.001

# 无序参数
epsilon_normal = 1.0
epsilon_strip = 1.0
strip_half_width = 0.1 / 2

# 文件保存
save_hdf5 = True  # 是否保存 HDF5 文件

print(f"✓ 参数设置完成")
print(f"  gamma = {gamma}")
print(f"  noise_strength = {noise_strength}")
print(f"  use_heat = {use_heat}")
print(f"  save_hdf5 = {save_hdf5}")

#%% ========== 创建 Layer ==========
print("\n" + "="*60)
print("创建 Layer")
print("="*60)

length_units = "um"

layer = tdgl.Layer(
    coherence_length=xi,
    london_lambda=london_lambda,
    thickness=d,
    gamma=gamma,
    u=u,
    conductivity=sigma,
    T_0=T_0,
    kappa_eff=kappa_eff,
    eta=hole_eta,
    C_eff=C_eff,
    T_heat=T_heat,
    suppress_electrode_edge_heating=suppress_electrode_edge_heating
)

print(f"✓ Layer 创建完成")

#%% ========== 创建几何 ==========
print("\n" + "="*60)
print("创建几何结构")
print("="*60)

extend = (length - electride_distance) / 2

# 创建薄膜
film = (
    tdgl.Polygon("film", points=box(width, length))
    .resample(401)
    .buffer(0.01)
)

# 探针点
probe_points = [(0, probe_distance), (0, -probe_distance)]

# 电极
terminal_width = width + 2
terminal_thickness = 1
top_terminal_y = length/2 - extend - terminal_thickness/2
bottom_terminal_y = -top_terminal_y

source = (
    tdgl.Polygon("source", points=box(terminal_width, terminal_thickness))
    .translate(dy=top_terminal_y)
)
drain = (
    tdgl.Polygon("drain", points=box(terminal_width, terminal_thickness))
    .translate(dy=bottom_terminal_y)
)

print(f"✓ 几何创建完成")
print(f"  Film: {width} x {length} um²")
print(f"  电极位置: y = ±{abs(top_terminal_y):.2f} um")
print(f"  延伸长度: {extend} um")

#%% ========== 创建 Device ==========
print("\n" + "="*60)
print("创建 Device")
print("="*60)

device = tdgl.Device(
    "debug_thermal_device",
    layer=layer,
    film=film,
    terminals=[source, drain],
    probe_points=probe_points,
    length_units=length_units,
)

print(f"✓ Device 创建完成: {device.name}")

#%% ========== 生成网格 ==========
print("\n" + "="*60)
print("生成网格")
print("="*60)

device.make_mesh(max_edge_length=xi/2, smooth=100)

print(f"✓ 网格生成完成")
print(f"  网格点数: {len(device.mesh.sites)}")
print(f"  三角形数: {len(device.mesh.elements)}")

#%% ========== 设置 eta 数组 ==========
print("\n" + "="*60)
print("设置热交换系数 eta")
print("="*60)

hole_gap_dimensionless = hole_gap / xi
y_coords = device.mesh.sites[:, 1]
mask = (y_coords >= -hole_gap_dimensionless) & (y_coords <= hole_gap_dimensionless)
eta_arr = np.full(len(device.mesh.sites), environment_eta)
eta_arr[mask] = hole_eta
device.layer.eta = eta_arr

print(f"✓ eta 设置完成")
print(f"  hole_gap (物理): {hole_gap} um")
print(f"  hole_gap (无量纲): {hole_gap_dimensionless} xi")
print(f"  孔区域点数: {np.sum(mask)} / {len(y_coords)} ({100*np.sum(mask)/len(y_coords):.1f}%)")

#%% ========== 可视化 Device ==========
print("\n" + "="*60)
print("可视化 Device")
print("="*60)

fig = plt.figure(figsize=(16, 10))

# 1. device.draw() - 彩色多边形
ax1 = plt.subplot(2, 3, 1)
device.draw(ax=ax1, legend=True, alpha=0.6)
ax1.set_title('Device Geometry - draw()', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(top_terminal_y, color='red', linestyle='--', alpha=0.7, lw=2)
ax1.axhline(bottom_terminal_y, color='red', linestyle='--', alpha=0.7, lw=2)
ax1.axhline(hole_gap, color='cyan', linestyle='-.', alpha=0.7, lw=2, label='Hole region')
ax1.axhline(-hole_gap, color='cyan', linestyle='-.', alpha=0.7, lw=2)

# 2. device.plot() with mesh
ax2 = plt.subplot(2, 3, 2)
device.plot(
    ax=ax2,
    legend=True,
    mesh=True,
    mesh_kwargs={'color': 'gray', 'lw': 0.3, 'alpha': 0.4}
)
ax2.set_title(f'Device + Mesh ({len(device.mesh.sites)} points)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. 网格放大（中心区域）
ax3 = plt.subplot(2, 3, 3)
x_mesh = device.mesh.sites[:, 0] * xi
y_mesh = device.mesh.sites[:, 1] * xi
# 绘制整个网格，但只显示中心区域
ax3.triplot(x_mesh, y_mesh, device.mesh.elements, 'k-', lw=0.5, alpha=0.3)
# 高亮中心区域的点
center_mask = (np.abs(x_mesh) < width/2) & (np.abs(y_mesh) < 5)
ax3.scatter(x_mesh[center_mask], y_mesh[center_mask], s=5, c='blue', alpha=0.8, zorder=3)
ax3.set_aspect('equal')
ax3.set_xlim(-width/2 - 0.5, width/2 + 0.5)
ax3.set_ylim(-5, 5)
ax3.set_title('Mesh Detail (center)', fontsize=12, fontweight='bold')
ax3.set_xlabel('x (μm)')
ax3.set_ylabel('y (μm)')
ax3.grid(True, alpha=0.3)
ax3.axhline(hole_gap, color='red', linestyle='--', alpha=0.7, lw=2, label='Hole region')
ax3.axhline(-hole_gap, color='red', linestyle='--', alpha=0.7, lw=2)
ax3.legend(fontsize=9)

# 4. eta 分布
ax4 = plt.subplot(2, 3, 4)
scatter = ax4.scatter(x_mesh, y_mesh, c=eta_arr, s=8, cmap='coolwarm', edgecolors='none')
ax4.set_aspect('equal')
ax4.set_title('Heat Exchange Coefficient η', fontsize=12, fontweight='bold')
ax4.set_xlabel('x (μm)')
ax4.set_ylabel('y (μm)')
plt.colorbar(scatter, ax=ax4, label='η')
ax4.axhline(hole_gap, color='white', linestyle='--', alpha=0.8, lw=2)
ax4.axhline(-hole_gap, color='white', linestyle='--', alpha=0.8, lw=2)
ax4.grid(True, alpha=0.3)

# 5. eta 沿 y 轴分布
ax5 = plt.subplot(2, 3, 5)
x_center_mask = np.abs(device.mesh.sites[:, 0]) < 0.1
y_center = device.mesh.sites[x_center_mask, 1] * xi
eta_center = eta_arr[x_center_mask]
sort_idx = np.argsort(y_center)
y_sorted = y_center[sort_idx]
eta_sorted = eta_center[sort_idx]

ax5.plot(y_sorted, eta_sorted, 'b-', linewidth=2.5, label='η(y)')
ax5.axvline(hole_gap, color='red', linestyle='--', alpha=0.6, lw=2, label=f'Hole ±{hole_gap} μm')
ax5.axvline(-hole_gap, color='red', linestyle='--', alpha=0.6, lw=2)
ax5.axhline(hole_eta, color='green', linestyle=':', alpha=0.6, lw=2, label=f'η_hole={hole_eta}')
ax5.axhline(environment_eta, color='orange', linestyle=':', alpha=0.6, lw=2, label=f'η_env={environment_eta}')
ax5.set_xlabel('y (μm)', fontsize=11)
ax5.set_ylabel('η', fontsize=11)
ax5.set_title('η Distribution (x≈0)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)

# 6. 参数摘要
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
param_text = f"""
参数摘要
{'='*40}

几何:
  尺寸: {width} × {length} μm²
  电极间距: {electride_distance} μm
  孔区域: |y| < {hole_gap} μm

材料:
  ξ = {xi} μm
  λ = {london_lambda} μm
  d = {d} μm
  γ = {gamma}
  u = {u}

热参数:
  T₀ = {T_0}
  κ_eff = {kappa_eff}
  C_eff = {C_eff}
  T_heat = {T_heat}
  η_hole = {hole_eta}
  η_env = {environment_eta}

求解:
  噪声强度 = {noise_strength}
  use_heat = {use_heat}
  网格点数 = {len(device.mesh.sites)}
"""
ax6.text(0.05, 0.95, param_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('debug_device.png', dpi=150, bbox_inches='tight')
print(f"✓ Device 可视化完成，已保存: debug_device.png")
plt.show()

#%% ========== 定义电流和无序函数 ==========
print("\n" + "="*60)
print("定义物理函数")
print("="*60)

def disorder_function(r, **kwargs):
    """定义无序参数 epsilon"""
    y = r[1]
    if -strip_half_width <= y <= strip_half_width:
        return epsilon_strip
    else:
        return epsilon_normal

solve_time = ramp_up_time + max_current_time + ramp_down_time + zero_current_time

def terminal_currents(t):
    """四阶段电流函数"""
    max_current = 3000 * d/0.08 * width/6

    t1 = ramp_up_time
    t2 = t1 + max_current_time
    t3 = t2 + ramp_down_time

    if t < t1:
        current = max_current * (t / ramp_up_time)
    elif t < t2:
        current = max_current
    elif t < t3:
        current = max_current * (1 - (t - t2) / ramp_down_time)
    else:
        current = 0

    return dict(source=current, drain=-current)

print(f"✓ 物理函数定义完成")
print(f"  总模拟时间: {solve_time}")
print(f"  最大电流: {3000 * d/0.08 * width/6:.2f}")

#%% ========== 生成文件名 ==========
print("\n" + "="*60)
print("生成文件名")
print("="*60)

# 根据非默认参数生成文件名
params_dict = {
    'gamma': (gamma, 1.0),
    'noise': (noise_strength, 0.0),
    'T_heat': (T_heat, 0.1),
    'hole_eta': (hole_eta, 0.05),
    'env_eta': (environment_eta, 10.0),
}

non_default = []
for key, (value, default) in params_dict.items():
    if value != default:
        non_default.append(f"{key}_{value}")

if non_default:
    filename_base = "_".join(non_default)
else:
    filename_base = "default_run"

if save_hdf5:
    output_filename = filename_base + ".h5"
    print(f"  HDF5 文件: {output_filename}")
else:
    output_filename = None
    print(f"  HDF5 文件: 不保存")

npz_filename = filename_base + ".npz"
print(f"  NPZ 文件: {npz_filename}")

#%% ========== 配置求解器 ==========
print("\n" + "="*60)
print("配置求解器")
print("="*60)

options = tdgl.SolverOptions(
    solve_time=solve_time,
    output_file=output_filename,
    field_units="mT",
    current_units="uA",
    dt_max=1e-2,
    save_every=1000,
    include_screening=False,
    max_solve_retries=100,
    adaptive=True,
    noise_strength=noise_strength,
    terminal_psi=None  # Let terminal regions evolve freely (not fixed to 0)
)

print(f"✓ 求解器配置完成")
print(f"  adaptive = {options.adaptive}")
print(f"  dt_max = {options.dt_max}")
print(f"  noise_strength = {options.noise_strength}")

#%% ========== 运行模拟 ==========
print("\n" + "="*60)
print("运行 TDGL 模拟")
print("="*60)

solution = tdgl.solve(
    device,
    options,
    applied_vector_potential=0,
    terminal_currents=terminal_currents,
    disorder_epsilon=disorder_function,
    use_heat=use_heat
)

print(f"✓ 模拟完成!")

#%% ========== 提取和保存数据 ==========
print("\n" + "="*60)
print("提取和保存数据")
print("="*60)

# 提取时间数据
time_data = solution.dynamics.time
print(f"  时间点数: {len(time_data)}")

# 计算电流
currents = np.array([terminal_currents(t)['source'] for t in time_data])

# 获取电压数据
voltage_data = solution.dynamics.mu

# 处理电压数据维度
if voltage_data.shape[0] == 2 and len(voltage_data.shape) == 2:
    voltage_data = voltage_data.T
    voltage_diff = voltage_data[:, 0] - voltage_data[:, 1]
elif len(voltage_data.shape) == 2 and voltage_data.shape[1] == 2:
    voltage_diff = voltage_data[:, 0] - voltage_data[:, 1]
else:
    probe_voltages = solution.dynamics.mu
    if len(probe_voltages) == 2:
        voltage_diff = probe_voltages[0] - probe_voltages[1]
    else:
        voltage_diff = probe_voltages[0]

# 转换到物理单位
try:
    V0 = device.V0()
    voltage_scale_uV = V0.to("uV").magnitude
    voltage_diff_uV = voltage_diff * voltage_scale_uV
    voltage_unit = "μV"
    print(f"  电压单位: {voltage_unit}")
except ValueError:
    voltage_diff_uV = voltage_diff
    voltage_unit = "a.u."
    print(f"  电压单位: {voltage_unit} (无量纲)")

# 确保数据长度匹配
min_length = min(len(currents), len(voltage_diff_uV))
currents = currents[:min_length]
voltage_diff_uV = voltage_diff_uV[:min_length]

# 保存 NPZ
np.savez(
    npz_filename,
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
    probe_distance=probe_distance,
    length=length,
    ramp_up_time=ramp_up_time,
    max_current_time=max_current_time,
    ramp_down_time=ramp_down_time,
    zero_current_time=zero_current_time,
    solve_time=solve_time,
    noise_strength=noise_strength
)

print(f"✓ 数据已保存到: {npz_filename}")
if save_hdf5 and output_filename:
    print(f"✓ HDF5 已保存到: {output_filename}")

#%% ========== 绘制 I-V 曲线 ==========
print("\n" + "="*60)
print("绘制 I-V 数据")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 电流 vs 时间
ax = axes[0, 0]
ax.plot(time_data[:min_length], currents, 'b-', linewidth=2)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Current (μA)', fontsize=12)
ax.set_title('Current vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# 2. 电压 vs 时间
ax = axes[0, 1]
ax.plot(time_data[:min_length], voltage_diff_uV, 'r-', linewidth=2)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel(f'Voltage ({voltage_unit})', fontsize=12)
ax.set_title('Voltage vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# 3. I-V 曲线
ax = axes[1, 0]
ax.plot(currents, voltage_diff_uV, 'g-', linewidth=2, alpha=0.7)
ax.scatter(currents[::100], voltage_diff_uV[::100], c=time_data[:min_length:100],
          cmap='viridis', s=30, zorder=3, edgecolors='black', linewidths=0.5)
ax.set_xlabel('Current (μA)', fontsize=12)
ax.set_ylabel(f'Voltage ({voltage_unit})', fontsize=12)
ax.set_title('I-V Curve (colored by time)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# 4. 功率 vs 时间
ax = axes[1, 1]
try:
    I0 = device.I0()
    current_scale_uA = I0.to("uA").magnitude
    currents_physical = currents * current_scale_uA
    power = currents_physical * voltage_diff_uV  # μW
    ax.plot(time_data[:min_length], power, 'm-', linewidth=2)
    ax.set_ylabel('Power (μW)', fontsize=12)
except:
    power = currents * voltage_diff_uV
    ax.plot(time_data[:min_length], power, 'm-', linewidth=2)
    ax.set_ylabel('Power (a.u.)', fontsize=12)

ax.set_xlabel('Time', fontsize=12)
ax.set_title('Power vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_iv_curves.png', dpi=150, bbox_inches='tight')
print(f"✓ I-V 曲线已保存: debug_iv_curves.png")
plt.show()

#%% ========== 完成 ==========
print("\n" + "="*60)
print("运行完成!")
print("="*60)
print(f"\n输出文件:")
print(f"  1. debug_device.png - Device 可视化")
print(f"  2. debug_iv_curves.png - I-V 曲线")
print(f"  3. {npz_filename} - 数据文件")
if save_hdf5 and output_filename:
    print(f"  4. {output_filename} - HDF5 完整数据")
print("\n" + "="*60)

# %%
