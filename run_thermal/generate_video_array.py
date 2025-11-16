"""
并行视频生成脚本 - SLURM数组任务版本
自动检测当前目录下的所有.h5文件，并根据SLURM_ARRAY_TASK_ID选择对应的文件
使用多进程并行生成帧，速度更快
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
import glob
import re

#%% ========== 参数设置 ==========
# 时间范围
TIME_START = 0
# TIME_END 将自动从HDF5文件检测 (默认10000，即ramp_up_time=5000 + ramp_down_time=5000)
# TIME_STEP = 10 (视频帧采样间隔，在主函数中设置)

# 输出设置
FPS = 10
DPI = 100
FIGSIZE = (16, 10)

# 并行设置 - 使用SLURM分配的CPU核心数
NUM_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', cpu_count()))

# 绘图选项
PLOT_ORDER = True          # |ψ| - 序参量幅度
PLOT_EPSILON = True        # ε - 无序参数
PLOT_TEMPERATURE = True    # T - 温度场 (计算为 1-ε)
PLOT_PHASE = True          # φ - 相位
PLOT_DPSI_DT = False       # |dψ/dt| - 时间导数（计算较慢）

# 颜色范围设置 (None = 自动)
ORDER_VMIN = 0.0           # |ψ| 最小值
ORDER_VMAX = 1.0           # |ψ| 最大值
EPSILON_VMIN = 0.0         # ε 最小值
EPSILON_VMAX = 1.0         # ε 最大值
TEMPERATURE_VMIN = 0.0     # T 最小值
TEMPERATURE_VMAX = 1.0     # T 最大值
PHASE_VMIN = -3.14159      # φ 最小值 (-π)
PHASE_VMAX = 3.14159       # φ 最大值 (+π)

# 物理参数
XI = 0.25  # coherence length (um)

#%% ========== 获取要处理的HDF5文件 ==========
def get_h5_files():
    """获取当前目录下所有.h5文件，按字母顺序排序"""
    h5_files = sorted(glob.glob("*.h5"))
    return h5_files

def get_current_h5_file():
    """根据SLURM_ARRAY_TASK_ID获取当前任务要处理的.h5文件"""
    h5_files = get_h5_files()

    if len(h5_files) == 0:
        raise ValueError("当前目录下没有找到.h5文件")

    # 从环境变量获取array task ID
    array_task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

    if array_task_id >= len(h5_files):
        raise ValueError(f"SLURM_ARRAY_TASK_ID ({array_task_id}) 超出.h5文件数量 ({len(h5_files)})")

    return h5_files[array_task_id]

def detect_save_every_from_filename(filename):
    """
    从文件名中检测save_every参数
    查找 'save_every_XXX' 模式，如果没有找到则返回默认值1000

    Example:
        'hole_gap_0.5_save_every_500.h5' -> 500
        'hole_gap_0.5.h5' -> 1000
    """
    # 使用正则表达式查找 save_every_数字 模式
    match = re.search(r'save_every_(\d+)', filename)
    if match:
        save_every = int(match.group(1))
        print(f"检测到 save_every 参数从文件名: {save_every}")
        return save_every
    else:
        print(f"文件名中未找到 save_every 参数，使用默认值: 1000")
        return 1000

def detect_time_parameters_from_hdf5(hdf_file):
    """
    从HDF5文件中检测时间参数（ramp_up_time, ramp_down_time等）

    Returns:
        dict: 包含检测到的时间参数，以及自动计算的TIME_END
    """
    import tdgl

    # 默认值
    params = {
        'ramp_up_time': 5000.0,
        'max_current_time': 0.0,
        'ramp_down_time': 5000.0,
        'zero_current_time': 0.0,
        'TIME_END': 10000.0,  # 默认值 (5000 + 0 + 5000 + 0)
        'auto_detected': False
    }

    try:
        solution = tdgl.Solution.from_hdf5(hdf_file)

        # 尝试从solution.options中获取solve_time
        if hasattr(solution, 'options') and hasattr(solution.options, 'solve_time'):
            solve_time = solution.options.solve_time
            params['TIME_END'] = solve_time
            print(f"从HDF5检测到 solve_time: {solve_time}")
            params['auto_detected'] = True

        # 如果无法从options获取，使用实际时间数据的最大值
        if not params['auto_detected']:
            time_data = solution.dynamics.time
            params['TIME_END'] = time_data.max()
            print(f"从HDF5数据检测到最大时间: {params['TIME_END']:.1f}")
            params['auto_detected'] = True

        # 尝试从文件名中检测各个时间参数
        filename = os.path.basename(hdf_file)

        # 检测 ramp_up_time
        match = re.search(r'ramp_up_time_([\d.]+)', filename)
        if match:
            params['ramp_up_time'] = float(match.group(1))
            print(f"从文件名检测到 ramp_up_time: {params['ramp_up_time']}")

        # 检测 max_current_time
        match = re.search(r'max_current_time_([\d.]+)', filename)
        if match:
            params['max_current_time'] = float(match.group(1))
            print(f"从文件名检测到 max_current_time: {params['max_current_time']}")

        # 检测 ramp_down_time
        match = re.search(r'ramp_down_time_([\d.]+)', filename)
        if match:
            params['ramp_down_time'] = float(match.group(1))
            print(f"从文件名检测到 ramp_down_time: {params['ramp_down_time']}")

        # 检测 zero_current_time
        match = re.search(r'zero_current_time_([\d.]+)', filename)
        if match:
            params['zero_current_time'] = float(match.group(1))
            print(f"从文件名检测到 zero_current_time: {params['zero_current_time']}")

    except Exception as e:
        print(f"警告: 无法从HDF5文件检测时间参数: {e}")
        print(f"使用默认值: TIME_END = {params['TIME_END']}")

    return params

#%% ========== 加载数据函数 ==========
def load_all_data(hdf_file):
    """在主进程中加载所有数据"""
    import tdgl
    print(f"\n加载数据...")
    print(f"tdgl路径: {tdgl.__file__}")

    solution = tdgl.Solution.from_hdf5(hdf_file)
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
        'hdf_file': hdf_file,  # 传递文件路径而不是数据
    }

#%% ========== 计算 dψ/dt 的函数 ==========
def calculate_dpsi_dt(hdf_file, t0, available_steps, time_data):
    """计算 dψ/dt - 按需从文件加载数据"""
    import tdgl

    # 找到最接近 t0 的 solve step
    time_diffs = [abs(time_data[available_steps.index(s)] - t0) for s in available_steps]
    step_idx = np.argmin(time_diffs)
    step_current = available_steps[step_idx]

    solution = tdgl.Solution.from_hdf5(hdf_file)

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
    (t0, frame_idx, data_dict, temp_dir) = args

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

    # 找到当前时间对应的数据
    idx = np.argmin(np.abs(time_data - t0))

    # 按需加载当前时间步的 psi 和 temperature
    solution = tdgl.Solution.from_hdf5(hdf_file)

    # 使用 tdgl 内置的方法找到最接近 t0 的 solve step
    step_current = solution.closest_solve_step(t0)
    solution.solve_step = step_current
    psi = solution.tdgl_data.psi.copy()

    # 动态加载 epsilon 和 temperature（如果HDF5中有的话）
    try:
        epsilon_current = solution.tdgl_data.epsilon.copy()
    except (AttributeError, KeyError):
        # 如果HDF5中没有epsilon数据，使用静态的epsilon_data
        epsilon_current = epsilon_data

    try:
        temperature_current = solution.tdgl_data.temperature.copy()
    except (AttributeError, KeyError):
        # 如果HDF5中没有temperature数据，计算为 1 - epsilon
        temperature_current = 1.0 - epsilon_current

    # 准备网格
    x = device_points[:, 0] / XI
    y = device_points[:, 1] / XI
    triangles = device_triangles

    # 计算子图数量
    plots_enabled = [PLOT_ORDER, PLOT_EPSILON, PLOT_TEMPERATURE, PLOT_PHASE, PLOT_DPSI_DT]
    num_field_plots = sum(plots_enabled)

    # 创建图形 - 使用GridSpec分为上下两部分
    # 上方：一行热图，下方：I-V曲线
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, num_field_plots, hspace=0.3, wspace=0.3,
                          height_ratios=[1, 0.6])

    plot_idx = 0

    # 绘制场分布（所有图在第一行）
    if PLOT_ORDER:
        ax = fig.add_subplot(gs[0, plot_idx])
        tc = ax.tripcolor(x, y, triangles, np.abs(psi), cmap='viridis',
                         vmin=ORDER_VMIN, vmax=ORDER_VMAX, shading='gouraud')
        ax.set_aspect('equal')
        ax.set_title(r'$|\psi|$ at t={:.1f}'.format(t0), fontsize=11)
        ax.set_xlabel(r'$x/\xi$')
        ax.set_ylabel(r'$y/\xi$')
        plt.colorbar(tc, ax=ax, label=r'$|\psi|$')
        plot_idx += 1

    if PLOT_EPSILON:
        ax = fig.add_subplot(gs[0, plot_idx])
        tc = ax.tripcolor(x, y, triangles, epsilon_current, cmap='RdBu_r',
                         vmin=EPSILON_VMIN, vmax=EPSILON_VMAX, shading='gouraud')
        ax.set_aspect('equal')
        ax.set_title(r'Disorder $\epsilon$ at t={:.1f}'.format(t0), fontsize=11)
        ax.set_xlabel(r'$x/\xi$')
        ax.set_ylabel(r'$y/\xi$')
        plt.colorbar(tc, ax=ax, label=r'$\epsilon$')
        plot_idx += 1

    if PLOT_TEMPERATURE:
        ax = fig.add_subplot(gs[0, plot_idx])
        tc = ax.tripcolor(x, y, triangles, temperature_current, cmap='hot',
                         vmin=TEMPERATURE_VMIN, vmax=TEMPERATURE_VMAX, shading='gouraud')
        ax.set_aspect('equal')
        ax.set_title(f'Temperature at t={t0:.1f}', fontsize=11)
        ax.set_xlabel(r'$x/\xi$')
        ax.set_ylabel(r'$y/\xi$')
        plt.colorbar(tc, ax=ax, label='T')
        plot_idx += 1

    if PLOT_PHASE:
        ax = fig.add_subplot(gs[0, plot_idx])
        phase = np.angle(psi)
        tc = ax.tripcolor(x, y, triangles, phase, cmap='twilight',
                         vmin=PHASE_VMIN, vmax=PHASE_VMAX, shading='gouraud')
        ax.set_aspect('equal')
        ax.set_title(r'Phase at t={:.1f}'.format(t0), fontsize=11)
        ax.set_xlabel(r'$x/\xi$')
        ax.set_ylabel(r'$y/\xi$')
        plt.colorbar(tc, ax=ax, label=r'$\phi$ (rad)')
        plot_idx += 1

    if PLOT_DPSI_DT:
        ax = fig.add_subplot(gs[0, plot_idx])
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
    # 创建一个横跨所有列的子图（第二行）
    ax_iv = fig.add_subplot(gs[1, :])

    # 绘制I-V曲线（横轴：电流，纵轴：电压）
    ax_iv.plot(currents_uA, voltage_diff_uV, 'k-', linewidth=2, alpha=0.7, label='I-V Curve')

    # 标记当前时间点
    ax_iv.scatter([currents_uA[idx]], [voltage_diff_uV[idx]], c='red', s=200, zorder=5,
                  edgecolors='white', linewidths=2.5, marker='o',
                  label=f't = {t0:.1f}')

    # 设置标签和标题
    ax_iv.set_xlabel(f'Current ({current_unit})', fontsize=12, fontweight='bold')
    ax_iv.set_ylabel(f'Voltage ({voltage_unit})', fontsize=12, fontweight='bold')
    ax_iv.set_title('I-V Curve', fontsize=13, fontweight='bold', pad=10)

    # 添加网格和图例
    ax_iv.grid(True, alpha=0.3, linestyle='--')
    ax_iv.legend(fontsize=11, loc='best')

    # 设置坐标轴范围
    ax_iv.set_xlim(currents_uA.min() * 0.95, currents_uA.max() * 1.05)
    ax_iv.set_ylim(voltage_diff_uV.min() * 0.95, voltage_diff_uV.max() * 1.05)

    # 保存帧
    frame_file = os.path.join(temp_dir, f"frame_{frame_idx:05d}.png")
    plt.savefig(frame_file, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

    return frame_file

#%% ========== 主函数 ==========
if __name__ == "__main__":
    # 获取当前任务要处理的.h5文件
    HDF_FILE = get_current_h5_file()

    # 从HDF5文件自动检测时间参数
    time_params = detect_time_parameters_from_hdf5(HDF_FILE)
    TIME_END = time_params['TIME_END']

    # TIME_STEP 默认为 10（视频帧采样间隔）
    TIME_STEP = 10

    # 从文件名检测save_every参数（HDF5文件中I-V曲线数据的原始采样间隔）
    IV_SAMPLING_STEP = detect_save_every_from_filename(HDF_FILE)

    # 生成输出文件名（去掉.h5扩展名，加上_video.mp4）
    OUTPUT_VIDEO = HDF_FILE.replace('.h5', '_video.mp4')
    TEMP_DIR = HDF_FILE.replace('.h5', '_temp_frames')

    print("="*60)
    print("并行视频生成 - SLURM数组任务")
    print("="*60)
    print(f"SLURM_ARRAY_TASK_ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 0)}")
    print(f"输入文件: {HDF_FILE}")
    print(f"视频帧时间步长: {TIME_STEP} (范围: {TIME_START} - {TIME_END})")
    print(f"I-V曲线数据采样间隔 (save_every): {IV_SAMPLING_STEP}")
    print(f"输出视频: {OUTPUT_VIDEO}")
    print(f"临时目录: {TEMP_DIR}")
    print(f"FPS: {FPS}, DPI: {DPI}")
    print(f"使用 {NUM_WORKERS} 个CPU核心进行并行处理")

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
    data_dict = load_all_data(HDF_FILE)

    # 生成时间点
    times = np.arange(TIME_START, TIME_END + TIME_STEP, TIME_STEP)
    times = times[times <= data_dict['time_data'].max()]

    print(f"\n将生成 {len(times)} 帧")

    # 创建临时目录
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"临时目录: {TEMP_DIR}")

    # 准备参数 - 每个worker传递整个data_dict
    args_list = [
        (t0, idx, data_dict, TEMP_DIR)
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
    print(f"输入文件: {HDF_FILE}")
    print(f"输出视频: {OUTPUT_VIDEO}")
    print(f"总帧数: {len(times)}")
    print(f"FPS: {FPS}")
    print(f"时长: {len(times)/FPS:.1f} 秒")
    print("="*60)
