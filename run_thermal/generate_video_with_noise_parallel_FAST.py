"""
并行视频生成脚本 - 支持 noise_strength 参数
使用多进程并行生成帧，速度更快
使用本地开发版本的 tdgl
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import h5py
import subprocess
import shutil

# 使用系统安装的 tdgl
# LOCAL_TDGL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'py-tdgl'))
# if LOCAL_TDGL_PATH not in sys.path:
#     sys.path.insert(0, LOCAL_TDGL_PATH)

#%% ========== 参数设置 ==========
# 输入文件
HDF_FILE = "noise_0.001.h5"

# 时间范围
TIME_START = 0
TIME_END = 8000
TIME_STEP = 10  # 时间步长

# 输出设置
OUTPUT_VIDEO = "noise_0.001-2_video.mp4"
TEMP_DIR = "frames_fast_noise_0.001-1"
FPS = 10
DPI = 100
FIGSIZE = (16, 10)

# 并行设置
NUM_WORKERS = None  # None = 使用所有核心

# 绘图选项
PLOT_ORDER = True
PLOT_EPSILON = False
PLOT_TEMPERATURE = True
PLOT_PHASE = True
PLOT_DPSI_DT = False

# 物理参数
XI = 0.25  # coherence length (um)

print("="*60)
print("并行视频生成")
print("="*60)
print(f"输入文件: {HDF_FILE}")
print(f"时间范围: {TIME_START} - {TIME_END}, 步长={TIME_STEP}")
print(f"输出视频: {OUTPUT_VIDEO}")
print(f"临时目录: {TEMP_DIR}")
print(f"FPS: {FPS}, DPI: {DPI}")

# 确定使用的CPU核心数
if NUM_WORKERS is None:
    NUM_WORKERS = cpu_count()
print(f"使用 {NUM_WORKERS} 个CPU核心进行并行处理")

#%% ========== 加载数据函数 ==========
def load_all_data():
    """在主进程中加载所有数据"""
    import tdgl
    print(f"\n加载数据...")
    print(f"tdgl路径: {tdgl.__file__}")

    solution = tdgl.Solution.from_hdf5(HDF_FILE)
    device = solution.device
    time_data = solution.dynamics.time

    # 获取可用的 solve steps
    with h5py.File(solution.path, 'r') as h5f:
        available_steps = sorted([int(k) for k in h5f['data'].keys()])

    # 获取电流和电压
    terminal_currents = solution.terminal_currents
    currents = np.array([terminal_currents(t)['source'] for t in time_data])

    try:
        I0 = device.I0()
        currents_uA = currents * I0.to("uA").magnitude
        current_unit = "μA"
    except:
        currents_uA = currents
        current_unit = "a.u."

    voltage_data = solution.dynamics.mu
    if voltage_data.shape[0] == 2 and len(voltage_data.shape) == 2:
        voltage_data = voltage_data.T
        voltage_diff = voltage_data[:, 0] - voltage_data[:, 1]
    elif len(voltage_data.shape) == 2 and voltage_data.shape[1] == 2:
        voltage_diff = voltage_data[:, 0] - voltage_data[:, 1]
    else:
        voltage_diff = voltage_data[0]

    try:
        V0 = device.V0()
        voltage_diff_uV = voltage_diff * V0.to("uV").magnitude
        voltage_unit = "μV"
    except:
        voltage_diff_uV = voltage_diff
        voltage_unit = "a.u."

    # 提取 device 的网格信息
    device_points = device.points.copy()
    device_triangles = device.triangles.copy()

    # 提取 epsilon 数据
    if callable(solution.disorder_epsilon):
        epsilon_data = np.array([solution.disorder_epsilon(site) for site in device.points])
    elif isinstance(solution.disorder_epsilon, (int, float)):
        epsilon_data = np.full(len(device.points), solution.disorder_epsilon)
    else:
        epsilon_data = solution.disorder_epsilon.copy() if hasattr(solution.disorder_epsilon, 'copy') else solution.disorder_epsilon

    # 不预加载数据，改为传递 HDF5 文件路径
    print(f"[OK] 元数据加载完成")
    print(f"  网格点数: {len(device_points)}")
    print(f"  时间点数: {len(time_data)}")
    print(f"  可用solve steps: {len(available_steps)}")
    print(f"  使用按需加载模式（节省内存）")

    return {
        'currents_uA': currents_uA,
        'voltage_diff_uV': voltage_diff_uV,
        'current_unit': current_unit,
        'voltage_unit': voltage_unit,
        'time_data': time_data,
        'available_steps': available_steps,
        'device_points': device_points,
        'device_triangles': device_triangles,
        'epsilon_data': epsilon_data,
        'hdf_file': HDF_FILE,  # 传递文件路径而不是数据
    }

#%% ========== 计算 d𝜓/dt 的函数 ==========
def calculate_dpsi_dt(hdf_file, t0, available_steps, time_data):
    """计算 d𝜓/dt - 按需从文件加载数据"""
    import tdgl

    solution = tdgl.Solution.from_hdf5(hdf_file)

    # 使用 solution.closest_solve_step 找到最接近的 solve step
    step_current = solution.closest_solve_step(t0)
    step_idx = available_steps.index(step_current)

    if step_idx > 0 and step_idx < len(available_steps) - 1:
        # 中心差分
        step_backward = available_steps[step_idx - 1]
        step_forward = available_steps[step_idx + 1]

        solution.solve_step = step_backward
        psi_backward = solution.tdgl_data.psi.copy()

        solution.solve_step = step_forward
        psi_forward = solution.tdgl_data.psi.copy()

        t_backward = time_data[available_steps.index(step_backward)]
        t_forward = time_data[available_steps.index(step_forward)]

        dt_total = t_forward - t_backward
        if dt_total > 0:
            dpsi_dt = (psi_forward - psi_backward) / dt_total
        else:
            dpsi_dt = np.zeros_like(psi_backward)

    elif step_idx == 0:
        # 前向差分
        step_forward = available_steps[1]

        solution.solve_step = step_current
        psi_current = solution.tdgl_data.psi.copy()

        solution.solve_step = step_forward
        psi_forward = solution.tdgl_data.psi.copy()

        t_current = time_data[0]
        t_forward = time_data[1]

        dt = t_forward - t_current
        if dt > 0:
            dpsi_dt = (psi_forward - psi_current) / dt
        else:
            dpsi_dt = np.zeros_like(psi_current)

    else:
        # 后向差分
        step_backward = available_steps[step_idx - 1]

        solution.solve_step = step_current
        psi_current = solution.tdgl_data.psi.copy()

        solution.solve_step = step_backward
        psi_backward = solution.tdgl_data.psi.copy()

        t_current = time_data[step_idx]
        t_backward = time_data[step_idx - 1]

        dt = t_current - t_backward
        if dt > 0:
            dpsi_dt = (psi_current - psi_backward) / dt
        else:
            dpsi_dt = np.zeros_like(psi_current)

    return np.abs(dpsi_dt)

#%% ========== 单帧绘制函数（worker进程） ==========
def plot_single_frame(args):
    """在worker进程中绘制单帧 - 按需加载数据"""
    (t0, frame_idx, data_dict) = args

    import matplotlib.pyplot as plt
    import numpy as np
    import tdgl
    import h5py

    # 解包数据
    currents_uA = data_dict['currents_uA']
    voltage_diff_uV = data_dict['voltage_diff_uV']
    current_unit = data_dict['current_unit']
    voltage_unit = data_dict['voltage_unit']
    time_data = data_dict['time_data']
    available_steps = data_dict['available_steps']
    device_points = data_dict['device_points']
    device_triangles = data_dict['device_triangles']
    epsilon_data = data_dict['epsilon_data']
    hdf_file = data_dict['hdf_file']

    # 找到当前时间对应的数据索引（用于IV曲线）
    idx = np.argmin(np.abs(time_data - t0))

    # 按需加载当前时间步的 psi 和 temperature
    solution = tdgl.Solution.from_hdf5(hdf_file)

    # 使用 solution.closest_solve_step 找到最接近的 solve step
    step_current = solution.closest_solve_step(t0)
    solution.solve_step = step_current
    psi = solution.tdgl_data.psi.copy()

    # Try to get temperature data
    # Method 1: Direct temperature field
    try:
        temperature = solution.tdgl_data.temperature.copy()
    except:
        # Method 2: Calculate from epsilon (main branch method: temperature = 1 - epsilon)
        try:
            epsilon = solution.tdgl_data.epsilon.copy()
            temperature = 1.0 - epsilon
        except:
            temperature = None

    # 准备网格
    x = device_points[:, 0] / XI
    y = device_points[:, 1] / XI
    triangles = device_triangles

    # 计算子图数量
    plots_enabled = [PLOT_ORDER, PLOT_EPSILON, PLOT_TEMPERATURE, PLOT_PHASE, PLOT_DPSI_DT]
    num_field_plots = sum(plots_enabled)
    ncols = 3
    field_nrows = (num_field_plots + ncols - 1) // ncols  # 场图的行数

    # 创建图形 - 使用GridSpec分为上下两部分
    fig = plt.figure(figsize=FIGSIZE)
    # 上方是场图网格，下方是I-V曲线（占整个宽度）
    gs = fig.add_gridspec(field_nrows + 1, ncols, hspace=0.4, wspace=0.3,
                          height_ratios=[1] * field_nrows + [0.8])

    plot_idx = 0

    # 绘制场分布
    if PLOT_ORDER:
        ax = fig.add_subplot(gs[plot_idx // ncols, plot_idx % ncols])
        tc = ax.tripcolor(x, y, triangles, np.abs(psi), cmap='viridis', shading='gouraud')
        ax.set_aspect('equal')
        ax.set_title(r'$|\psi|$ at t={:.1f}'.format(t0), fontsize=11)
        ax.set_xlabel(r'$x/\xi$')
        ax.set_ylabel(r'$y/\xi$')
        plt.colorbar(tc, ax=ax, label=r'$|\psi|$')
        plot_idx += 1

    if PLOT_EPSILON:
        ax = fig.add_subplot(gs[plot_idx // ncols, plot_idx % ncols])
        tc = ax.tripcolor(x, y, triangles, epsilon_data, cmap='RdBu_r',
                         vmin=-1.0, vmax=1.0, shading='gouraud')
        ax.set_aspect('equal')
        ax.set_title(r'Disorder $\epsilon$', fontsize=11)
        ax.set_xlabel(r'$x/\xi$')
        ax.set_ylabel(r'$y/\xi$')
        plt.colorbar(tc, ax=ax, label=r'$\epsilon$')
        plot_idx += 1

    if PLOT_TEMPERATURE:
        ax = fig.add_subplot(gs[plot_idx // ncols, plot_idx % ncols])
        if temperature is not None:
            tc = ax.tripcolor(x, y, triangles, temperature, cmap='hot',
                            vmin=0.0, vmax=2.0, shading='gouraud')
            ax.set_aspect('equal')
            ax.set_title(f'Temperature at t={t0:.1f}', fontsize=11)
            ax.set_xlabel(r'$x/\xi$')
            ax.set_ylabel(r'$y/\xi$')
            plt.colorbar(tc, ax=ax, label='T')
        else:
            ax.text(0.5, 0.5, 'Temperature\nNot Available',
                   ha='center', va='center', transform=ax.transAxes)
        plot_idx += 1

    if PLOT_PHASE:
        ax = fig.add_subplot(gs[plot_idx // ncols, plot_idx % ncols])
        phase = np.angle(psi)
        tc = ax.tripcolor(x, y, triangles, phase, cmap='twilight',
                         vmin=-np.pi, vmax=np.pi, shading='gouraud')
        ax.set_aspect('equal')
        ax.set_title(r'Phase at t={:.1f}'.format(t0), fontsize=11)
        ax.set_xlabel(r'$x/\xi$')
        ax.set_ylabel(r'$y/\xi$')
        plt.colorbar(tc, ax=ax, label=r'$\phi$ (rad)')
        plot_idx += 1

    if PLOT_DPSI_DT:
        ax = fig.add_subplot(gs[plot_idx // ncols, plot_idx % ncols])
        dpsi_dt_magnitude = calculate_dpsi_dt(hdf_file, t0, available_steps, time_data)
        if dpsi_dt_magnitude is not None and dpsi_dt_magnitude.max() > 0:
            vmax = np.percentile(dpsi_dt_magnitude, 99)
            tc = ax.tripcolor(x, y, triangles, dpsi_dt_magnitude,
                            cmap='plasma', vmin=0, vmax=vmax, shading='gouraud')
            ax.set_aspect('equal')
            ax.set_title(r'$|d\psi/dt|$ at t={:.1f}'.format(t0), fontsize=11)
            ax.set_xlabel(r'$x/\xi$')
            ax.set_ylabel(r'$y/\xi$')
            plt.colorbar(tc, ax=ax, label=r'$|d\psi/dt|$')
        else:
            ax.text(0.5, 0.5, r'$|d\psi/dt|$' + '\nNot Available',
                   ha='center', va='center', transform=ax.transAxes)
        plot_idx += 1

    # I-V 曲线 - 底部横跨整个宽度
    # 创建一个横跨所有列的子图
    ax_iv = fig.add_subplot(gs[field_nrows, :])

    # 创建双y轴
    ax1 = ax_iv
    ax2 = ax1.twinx()

    # 绘制电流曲线（左y轴，蓝色）
    line1 = ax1.plot(time_data, currents_uA, 'b-', linewidth=1.5, alpha=0.6, label='Current')
    point1 = ax1.scatter([t0], [currents_uA[idx]], c='blue', s=150, zorder=5,
                         edgecolors='white', linewidths=2, marker='o')
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'Current ({current_unit})', fontsize=11, color='blue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 绘制电压曲线（右y轴，红色）
    line2 = ax2.plot(time_data, voltage_diff_uV, 'r-', linewidth=1.5, alpha=0.6, label='Voltage')
    point2 = ax2.scatter([t0], [voltage_diff_uV[idx]], c='red', s=150, zorder=5,
                         edgecolors='white', linewidths=2, marker='o')
    ax2.set_ylabel(f'Voltage ({voltage_unit})', fontsize=11, color='red', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')

    # 添加网格和标题
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title(f'Current-Voltage vs Time (t = {t0:.1f})', fontsize=13, fontweight='bold', pad=10)

    # 添加垂直线标记当前时间
    ax1.axvline(x=t0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # 设置x轴范围，确保显示所有数据
    ax1.set_xlim(time_data.min(), time_data.max())

    # 保存帧
    frame_file = os.path.join(TEMP_DIR, f"frame_{frame_idx:05d}.png")
    plt.savefig(frame_file, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    return frame_file

#%% ========== 主函数 ==========
if __name__ == "__main__":
    # 首先处理 noise_strength 参数 - 永久删除以避免兼容性问题
    temp_hdf_file = None
    with h5py.File(HDF_FILE, 'r') as f:
        has_noise_strength = 'noise_strength' in f['solution/options'].attrs

    if has_noise_strength:
        print(f"\n检测到不兼容的 noise_strength 参数，将从文件中删除...")
        with h5py.File(HDF_FILE, 'r+') as f:
            del f['solution/options'].attrs['noise_strength']
        print(f"[OK] 已删除 noise_strength 参数\n")

    # 加载所有数据
    data_dict = load_all_data()

    # 生成时间点
    times = np.arange(TIME_START, TIME_END + TIME_STEP, TIME_STEP)
    times = times[times <= data_dict['time_data'].max()]

    print(f"\n将生成 {len(times)} 帧")

    # 创建临时目录
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"临时目录: {TEMP_DIR}")

    # 准备参数 - 每个worker传递整个data_dict
    args_list = [
        (t0, idx, data_dict)
        for idx, t0 in enumerate(times)
    ]

    # 并行生成帧
    print(f"\n并行生成帧 (使用 {NUM_WORKERS} 个核心)...")
    with Pool(NUM_WORKERS) as pool:
        frame_files = list(tqdm(
            pool.imap(plot_single_frame, args_list),
            total=len(args_list),
            desc="生成帧",
            unit="frame"
        ))

    print(f"\n[OK] 所有帧已生成")

    # 使用 ffmpeg 合成视频
    print(f"\n合成视频...")
    frame_pattern = os.path.join(TEMP_DIR, "frame_%05d.png")

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', frame_pattern,
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # 确保尺寸能被2整除
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        OUTPUT_VIDEO
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"[OK] 视频已保存: {OUTPUT_VIDEO}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg错误: {e.stderr.decode()}")
        print("请确保已安装 ffmpeg")

    # 清理临时文件
    print(f"\n清理临时文件...")
    try:
        shutil.rmtree(TEMP_DIR)
        print(f"[OK] 临时目录已删除")
    except Exception as e:
        print(f"警告: 无法删除临时目录: {e}")

    # 总结
    print("\n" + "="*60)
    print("完成!")
    print("="*60)
    print(f"输出视频: {OUTPUT_VIDEO}")
    print(f"总帧数: {len(times)}")
    print(f"FPS: {FPS}")
    print(f"时长: {len(times)/FPS:.1f} 秒")
    print("="*60)
