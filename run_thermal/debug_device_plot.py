#%%
"""
调试脚本 - 用于可视化device几何结构
参数在脚本内部设置，方便快速调试
"""
import numpy as np
import matplotlib.pyplot as plt
import tdgl
from tdgl.geometry import box

#%% 参数设置 - 在这里修改所有参数
# 材料参数
xi = 0.25  # coherence length (um)
london_lambda = 0.25  # London penetration depth (um)
d = 0.01  # thickness (um)
gamma = 1.0
u = 5.79
sigma = 1e3

# 热参数
T_0 = 1.0  # 无量纲临界温度
kappa_eff = 0.02  # 有效热导率
hole_eta = 0.05  # 孔区域与环境的热交换系数
environment_eta = 10.0  # 其他区域与环境的热交换系数
C_eff = 1.0  # 有效热容
T_heat = 0.1
suppress_electrode_edge_heating = True

# 几何参数
width = 6.0  # 总宽度 (um)
length = 18.0  # 总长度 (um)
electride_distance = 8.0  # 电极之间的距离 (um)
probe_distance = 2.5  # 探针位置 (um)
hole_gap = 1.0  # 孔区域的大小 (um)

# 网格参数
max_edge_length = xi / 2  # 最大网格边长
smooth = 100  # 网格平滑次数

# 无序参数
epsilon_normal = 1.0  # 正常超导区域的epsilon值
epsilon_strip = 1.0  # 中间条带区域的epsilon值
strip_half_width = 0.1 / 2  # 中间条带的半宽度

#%% 创建Layer
print("="*60)
print("创建Layer...")
print("="*60)

layer = tdgl.Layer(
    coherence_length=xi,
    london_lambda=london_lambda,
    thickness=d,
    gamma=gamma,
    u=u,
    conductivity=sigma,
    T_0=T_0,
    kappa_eff=kappa_eff,
    eta=hole_eta,  # 初始值，稍后会修改
    C_eff=C_eff,
    T_heat=T_heat
)

print(f"✓ Layer created with:")
print(f"  xi = {xi} um")
print(f"  lambda = {london_lambda} um")
print(f"  d = {d} um")
print(f"  gamma = {gamma}")
print(f"  u = {u}")

#%% 创建几何结构
print("\n" + "="*60)
print("创建几何结构...")
print("="*60)

# 计算延伸长度
extend = (length - electride_distance) / 2

print(f"总长度: {length} um")
print(f"电极间距: {electride_distance} um")
print(f"延伸长度: {extend} um")

# 创建超导薄膜
film = (
    tdgl.Polygon("film", points=box(width, length))
    .resample(401)
    .buffer(0.01)
)

print(f"✓ Film created: {width} x {length} um²")

# 创建探针点
probe_points = [(0, probe_distance), (0, -probe_distance)]
print(f"✓ Probe points: {probe_points}")

# 创建电极
terminal_width = width + 2  # 电极宽度
terminal_thickness = 1  # 电极厚度

# 计算电极位置
top_terminal_y = length/2 - extend - terminal_thickness/2
bottom_terminal_y = -top_terminal_y

print(f"\n电极参数:")
print(f"  宽度: {terminal_width} um")
print(f"  厚度: {terminal_thickness} um")
print(f"  顶部电极 y 坐标: {top_terminal_y} um")
print(f"  底部电极 y 坐标: {bottom_terminal_y} um")

# 创建电流端子
source = (
    tdgl.Polygon("source", points=box(terminal_width, terminal_thickness))
    .translate(dy=top_terminal_y)
)
drain = (
    tdgl.Polygon("drain", points=box(terminal_width, terminal_thickness))
    .translate(dy=bottom_terminal_y)
)

print(f"✓ Terminals created")

#%% 创建Device
print("\n" + "="*60)
print("创建Device...")
print("="*60)

device = tdgl.Device(
    "debug_device",
    layer=layer,
    film=film,
    terminals=[source, drain],
    probe_points=probe_points,
    length_units="um",
)

print(f"✓ Device created: {device.name}")

#%% 生成网格
print("\n" + "="*60)
print("生成网格...")
print("="*60)

device.make_mesh(max_edge_length=max_edge_length, smooth=smooth)

print(f"✓ Mesh generated:")
print(f"  网格点数: {len(device.mesh.sites)}")
print(f"  三角形数: {len(device.mesh.elements)}")
print(f"  最大边长: {max_edge_length} um")

#%% 设置eta数组（热交换系数）
print("\n" + "="*60)
print("设置热交换系数...")
print("="*60)

# 将hole_gap从物理单位转换为无量纲单位
hole_gap_dimensionless = hole_gap / xi

y_coords = device.mesh.sites[:, 1]  # 无量纲坐标（单位为xi）
mask = (y_coords >= -hole_gap_dimensionless) & (y_coords <= hole_gap_dimensionless)
eta_arr = np.full(len(device.mesh.sites), environment_eta)
eta_arr[mask] = hole_eta
device.layer.eta = eta_arr

print(f"hole_gap (物理): {hole_gap} um")
print(f"hole_gap (无量纲): {hole_gap_dimensionless} xi")
print(f"孔区域网格点数: {np.sum(mask)} / {len(y_coords)} ({100*np.sum(mask)/len(y_coords):.1f}%)")
print(f"环境区域 eta: {environment_eta}")
print(f"孔区域 eta: {hole_eta}")

#%% 绘制Device
print("\n" + "="*60)
print("绘制Device...")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 绘制整体几何结构
ax = axes[0, 0]
device.plot(ax=ax, show_sites=False)
ax.set_title('Device Geometry (整体结构)', fontsize=14, fontweight='bold')
ax.set_xlabel('x (um)')
ax.set_ylabel('y (um)')
ax.grid(True, alpha=0.3)

# 添加标注
ax.axhline(top_terminal_y, color='red', linestyle='--', alpha=0.5, label='Top electrode')
ax.axhline(bottom_terminal_y, color='red', linestyle='--', alpha=0.5, label='Bottom electrode')
ax.axhline(probe_distance, color='green', linestyle=':', alpha=0.5, label='Top probe')
ax.axhline(-probe_distance, color='green', linestyle=':', alpha=0.5, label='Bottom probe')
ax.axhline(hole_gap_dimensionless * xi, color='blue', linestyle='-.', alpha=0.5, label='Hole boundary')
ax.axhline(-hole_gap_dimensionless * xi, color='blue', linestyle='-.', alpha=0.5)
ax.legend()

# 2. 绘制网格
ax = axes[0, 1]
x = device.mesh.sites[:, 0] * xi
y = device.mesh.sites[:, 1] * xi
ax.triplot(x, y, device.mesh.elements, 'k-', lw=0.3, alpha=0.3)
ax.scatter(x, y, s=1, c='blue', alpha=0.5)
ax.set_aspect('equal')
ax.set_title(f'Mesh ({len(device.mesh.sites)} points)', fontsize=14, fontweight='bold')
ax.set_xlabel('x (um)')
ax.set_ylabel('y (um)')
ax.grid(True, alpha=0.3)

# 3. 绘制热交换系数分布
ax = axes[1, 0]
x_mesh = device.mesh.sites[:, 0] * xi
y_mesh = device.mesh.sites[:, 1] * xi
scatter = ax.scatter(x_mesh, y_mesh, c=eta_arr, s=5, cmap='coolwarm')
ax.set_aspect('equal')
ax.set_title('Heat Exchange Coefficient η', fontsize=14, fontweight='bold')
ax.set_xlabel('x (um)')
ax.set_ylabel('y (um)')
plt.colorbar(scatter, ax=ax, label='η')
ax.axhline(hole_gap_dimensionless * xi, color='white', linestyle='--', alpha=0.7, lw=2)
ax.axhline(-hole_gap_dimensionless * xi, color='white', linestyle='--', alpha=0.7, lw=2)
ax.grid(True, alpha=0.3)

# 4. 绘制y方向的eta分布
ax = axes[1, 1]
# 沿着x=0的直线提取eta值
x_center_mask = np.abs(device.mesh.sites[:, 0]) < 0.1  # 选择x接近0的点
y_center = device.mesh.sites[x_center_mask, 1] * xi
eta_center = eta_arr[x_center_mask]

# 排序以便绘图
sort_idx = np.argsort(y_center)
y_sorted = y_center[sort_idx]
eta_sorted = eta_center[sort_idx]

ax.plot(y_sorted, eta_sorted, 'b-', linewidth=2)
ax.axvline(hole_gap, color='red', linestyle='--', alpha=0.5, label=f'Hole boundary (±{hole_gap} um)')
ax.axvline(-hole_gap, color='red', linestyle='--', alpha=0.5)
ax.axhline(hole_eta, color='green', linestyle=':', alpha=0.5, label=f'Hole η = {hole_eta}')
ax.axhline(environment_eta, color='orange', linestyle=':', alpha=0.5, label=f'Environment η = {environment_eta}')
ax.set_xlabel('y (um)', fontsize=12)
ax.set_ylabel('η', fontsize=12)
ax.set_title('η Distribution along y-axis (x≈0)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('debug_device_plot.png', dpi=150, bbox_inches='tight')
print(f"✓ 图像已保存: debug_device_plot.png")
plt.show()

#%% 打印总结信息
print("\n" + "="*60)
print("Device 创建完成!")
print("="*60)
print(f"\n几何参数:")
print(f"  薄膜尺寸: {width} x {length} um²")
print(f"  电极位置: y = ±{abs(top_terminal_y):.2f} um")
print(f"  电极间距: {electride_distance} um")
print(f"  延伸长度: {extend} um")
print(f"  探针位置: y = ±{probe_distance} um")
print(f"  孔区域: |y| < {hole_gap} um")

print(f"\n网格信息:")
print(f"  网格点数: {len(device.mesh.sites)}")
print(f"  三角形数: {len(device.mesh.elements)}")
print(f"  最大边长: {max_edge_length} um")

print(f"\n材料参数:")
print(f"  ξ = {xi} um")
print(f"  λ = {london_lambda} um")
print(f"  d = {d} um")
print(f"  γ = {gamma}")
print(f"  u = {u}")
print(f"  σ = {sigma}")

print(f"\n热参数:")
print(f"  T_0 = {T_0}")
print(f"  κ_eff = {kappa_eff}")
print(f"  C_eff = {C_eff}")
print(f"  T_heat = {T_heat}")
print(f"  环境 η = {environment_eta}")
print(f"  孔区域 η = {hole_eta}")
print(f"  suppress_electrode_edge_heating = {suppress_electrode_edge_heating}")

print("\n" + "="*60)

# %%
