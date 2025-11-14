#%%
"""
调试脚本 - 使用官方API可视化device几何结构
基于 https://py-tdgl.readthedocs.io/en/latest/api/device.html
"""
import numpy as np
import matplotlib.pyplot as plt
import tdgl
from tdgl.geometry import box

#%% 参数设置
# 材料参数
xi = 0.25  # coherence length (um)
london_lambda = 0.25  # London penetration depth (um)
d = 0.01  # thickness (um)
gamma = 1.0
u = 5.79
sigma = 1e3

# 热参数
T_0 = 1.0
kappa_eff = 0.02
hole_eta = 0.05
environment_eta = 10.0
C_eff = 1.0
T_heat = 0.1

# 几何参数
width = 6.0  # um
length = 18.0  # um
electride_distance = 18.5  # um
probe_distance = 2.5  # um
hole_gap = 1.0  # um

# 网格参数
max_edge_length = xi / 2
smooth = 100

#%% 创建Layer
print("="*60)
print("创建Device...")
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
    eta=hole_eta,
    C_eff=C_eff,
    T_heat=T_heat
)

#%% 创建几何
extend = (length - electride_distance) / 2

# 薄膜
film = (
    tdgl.Polygon("film", points=box(width, length))
    .resample(401)
    .buffer(0.01)
)

# 探针
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

#%% 创建Device
device = tdgl.Device(
    "thermal_debug",
    layer=layer,
    film=film,
    terminals=[source, drain],
    probe_points=probe_points,
    length_units="um",
)

print(f"✓ Device创建完成")

#%% 生成网格
device.make_mesh(max_edge_length=max_edge_length, smooth=smooth)
print(f"✓ 网格生成完成: {len(device.mesh.sites)} 个点")

#%% 设置eta数组
hole_gap_dimensionless = hole_gap / xi
y_coords = device.mesh.sites[:, 1]
mask = (y_coords >= -hole_gap_dimensionless) & (y_coords <= hole_gap_dimensionless)
eta_arr = np.full(len(device.mesh.sites), environment_eta)
eta_arr[mask] = hole_eta
device.layer.eta = eta_arr

print(f"✓ 热交换系数设置完成")
print(f"  孔区域点数: {np.sum(mask)} ({100*np.sum(mask)/len(y_coords):.1f}%)")

#%% 可视化 - 使用官方API
print("\n" + "="*60)
print("使用官方API绘制Device...")
print("="*60)

fig = plt.figure(figsize=(16, 12))

# 1. device.draw() - 填充的多边形patches
ax1 = plt.subplot(2, 3, 1)
device.draw(ax=ax1, legend=True, alpha=0.6)
ax1.set_title('device.draw() - 填充多边形\n(alpha=0.6)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. device.draw() with exclude - 排除terminals
ax2 = plt.subplot(2, 3, 2)
device.draw(ax=ax2, legend=True, alpha=0.7, exclude=['source', 'drain'])
ax2.set_title('device.draw(exclude=[terminals])\n仅显示film', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. device.plot() - 边界线条
ax3 = plt.subplot(2, 3, 3)
device.plot(ax=ax3, legend=True, mesh=False, linewidth=2, color='blue')
ax3.set_title('device.plot() - 边界线\n(mesh=False)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. device.plot() with mesh - 边界+网格
ax4 = plt.subplot(2, 3, 4)
device.plot(
    ax=ax4,
    legend=True,
    mesh=True,
    mesh_kwargs={'color': 'gray', 'lw': 0.3, 'alpha': 0.4}
)
ax4.set_title(f'device.plot(mesh=True)\n{len(device.mesh.sites)} 网格点',
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. 热交换系数分布
ax5 = plt.subplot(2, 3, 5)
x_mesh = device.mesh.sites[:, 0] * xi
y_mesh = device.mesh.sites[:, 1] * xi
scatter = ax5.scatter(x_mesh, y_mesh, c=eta_arr, s=8, cmap='coolwarm', edgecolors='none')
ax5.set_aspect('equal')
ax5.set_title('热交换系数 η 分布', fontsize=12, fontweight='bold')
ax5.set_xlabel('x (μm)')
ax5.set_ylabel('y (μm)')
plt.colorbar(scatter, ax=ax5, label='η')
ax5.axhline(hole_gap, color='white', linestyle='--', alpha=0.8, lw=2, label='孔边界')
ax5.axhline(-hole_gap, color='white', linestyle='--', alpha=0.8, lw=2)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. η沿y轴的分布
ax6 = plt.subplot(2, 3, 6)
x_center_mask = np.abs(device.mesh.sites[:, 0]) < 0.1
y_center = device.mesh.sites[x_center_mask, 1] * xi
eta_center = eta_arr[x_center_mask]
sort_idx = np.argsort(y_center)
y_sorted = y_center[sort_idx]
eta_sorted = eta_center[sort_idx]

ax6.plot(y_sorted, eta_sorted, 'b-', linewidth=2.5, label='η(y)')
ax6.axvline(hole_gap, color='red', linestyle='--', alpha=0.6, lw=2, label=f'孔边界 (±{hole_gap} μm)')
ax6.axvline(-hole_gap, color='red', linestyle='--', alpha=0.6, lw=2)
ax6.axhline(hole_eta, color='green', linestyle=':', alpha=0.6, lw=2, label=f'孔 η={hole_eta}')
ax6.axhline(environment_eta, color='orange', linestyle=':', alpha=0.6, lw=2, label=f'环境 η={environment_eta}')
ax6.set_xlabel('y (μm)', fontsize=11)
ax6.set_ylabel('η', fontsize=11)
ax6.set_title('η 沿 y 轴分布 (x≈0)', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=9)

plt.tight_layout()
plt.savefig('debug_device_visualize.png', dpi=150, bbox_inches='tight')
print(f"✓ 图像已保存: debug_device_visualize.png")
plt.show()

#%% 打印设备信息
print("\n" + "="*60)
print("Device 信息总结")
print("="*60)

print(f"\n几何参数:")
print(f"  Film: {width} × {length} μm²")
print(f"  电极位置: y = ±{abs(top_terminal_y):.2f} μm")
print(f"  电极尺寸: {terminal_width} × {terminal_thickness} μm²")
print(f"  电极间距: {electride_distance} μm")
print(f"  延伸长度: {extend} μm")
print(f"  探针: y = ±{probe_distance} μm")
print(f"  孔区域: |y| < {hole_gap} μm")

print(f"\n网格:")
print(f"  点数: {len(device.mesh.sites)}")
print(f"  三角形: {len(device.mesh.elements)}")
print(f"  最大边长: {max_edge_length} μm = ξ/2")

print(f"\n材料参数:")
print(f"  ξ = {xi} μm")
print(f"  λ = {london_lambda} μm")
print(f"  d = {d} μm")
print(f"  γ = {gamma}")
print(f"  u = {u}")
print(f"  σ = {sigma}")

print(f"\n热参数:")
print(f"  T₀ = {T_0}")
print(f"  κ_eff = {kappa_eff}")
print(f"  C_eff = {C_eff}")
print(f"  T_heat = {T_heat}")
print(f"  孔区域 η = {hole_eta}")
print(f"  环境 η = {environment_eta}")

print("\n" + "="*60)
print("可视化方法说明:")
print("="*60)
print("\n1. device.draw(alpha=0.6)")
print("   - 绘制填充的彩色多边形patches")
print("   - alpha控制透明度 (0-1)")
print("   - 适合展示几何结构")

print("\n2. device.draw(exclude=['source', 'drain'])")
print("   - 可以排除指定的多边形")
print("   - 适合聚焦特定区域")

print("\n3. device.plot(mesh=False)")
print("   - 绘制多边形边界线")
print("   - 可自定义线条颜色、宽度")
print("   - 自动显示probe points")

print("\n4. device.plot(mesh=True, mesh_kwargs={...})")
print("   - 同时显示边界和三角网格")
print("   - mesh_kwargs控制网格样式")
print("   - 适合检查网格质量")

print("\n" + "="*60)

# %%
