# run_tdgl_with_different_gamma.py 使用说明

## 功能说明

此脚本会自动检测与默认值不同的参数，并根据这些参数生成文件名。

## 文件命名规则

**格式**: `参数名_值`，多个参数用下划线连接

**示例**:
- 如果只修改了 `gamma=2.0`，文件名为: `gamma_2.0.h5` 和 `gamma_2.0.npz`
- 如果修改了多个参数，例如 `gamma=2.0` 和 `noise_strength=0.1`，文件名为: `gamma_2.0_noise_strength_0.1.h5` 和 `gamma_2.0_noise_strength_0.1.npz`
- 如果所有参数都是默认值，文件名为: `default_run.h5` 和 `default_run.npz`

## 基本用法

### 1. 使用默认参数（不保存HDF5）
```bash
python run_tdgl_with_different_gamma.py
```
- 只保存 `default_run.npz`
- 不保存 HDF5 文件

### 2. 使用默认参数（保存HDF5）
```bash
python run_tdgl_with_different_gamma.py --save_hdf5
```
- 保存 `default_run.h5` 和 `default_run.npz`

### 3. 修改单个参数
```bash
python run_tdgl_with_different_gamma.py --gamma 2.0 --save_hdf5
```
- 保存 `gamma_2.0.h5` 和 `gamma_2.0.npz`

### 4. 修改多个参数
```bash
python run_tdgl_with_different_gamma.py --gamma 2.0 --noise_strength 0.1 --T_heat 0.2 --save_hdf5
```
- 保存 `gamma_2.0_noise_strength_0.1_T_heat_0.2.h5` 和 `gamma_2.0_noise_strength_0.1_T_heat_0.2.npz`

## 默认参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gamma` | 1.0 | Ginzburg-Landau 参数 |
| `hole_eta` | 0.05 | 孔区域热交换系数 |
| `environment_eta` | 10.0 | 环境热交换系数 |
| `C_eff` | 1.0 | 有效热容 |
| `T_heat` | 0.1 | 热温度 |
| `T_0` | 1.0 | 临界温度 |
| `kappa_eff` | 0.02 | 有效热导率 |
| `hole_gap` | 1.0 | 孔区域大小 (μm) |
| `d` | 0.01 | 厚度 (μm) |
| `xi` | 0.25 | 相干长度 (μm) |
| `london_lambda` | 0.25 | London 穿透深度 (μm) |
| `sigma` | 1000 | 电导率 |
| `use_heat` | True | 是否使用热方程 |
| `u` | 5.79 | u 参数 |
| `width` | 6.0 | 宽度 (μm) |
| `electride_distance` | 18.5 | 电极间距 (μm) |
| `probe_distance` | 2.5 | 探针距离 (μm) |
| `length` | 18.0 | 长度 (μm) |
| `ramp_up_time` | 5000 | 电流上升时间 |
| `max_current_time` | 0 | 最大电流保持时间 |
| `ramp_down_time` | 5000 | 电流下降时间 |
| `zero_current_time` | 0 | 零电流保持时间 |
| `suppress_electrode_edge_heating` | True | 抑制电极边缘加热 |
| `noise_strength` | 0.0 | 随机噪声强度 |

## 实用示例

### 测试不同的 gamma 值
```bash
# gamma = 0.5
python run_tdgl_with_different_gamma.py --gamma 0.5 --save_hdf5

# gamma = 1.0 (默认，如果单独运行)
python run_tdgl_with_different_gamma.py --save_hdf5  # 文件名: default_run.h5

# gamma = 2.0
python run_tdgl_with_different_gamma.py --gamma 2.0 --save_hdf5
```

### 添加噪声模拟
```bash
# 弱噪声
python run_tdgl_with_different_gamma.py --noise_strength 0.01 --save_hdf5

# 中等噪声
python run_tdgl_with_different_gamma.py --noise_strength 0.1 --save_hdf5

# 强噪声
python run_tdgl_with_different_gamma.py --noise_strength 0.5 --save_hdf5
```

### 改变热参数
```bash
python run_tdgl_with_different_gamma.py --T_heat 0.2 --kappa_eff 0.05 --save_hdf5
# 文件名: T_heat_0.2_kappa_eff_0.05.h5
```

### 组合研究
```bash
# 研究噪声和gamma的组合效应
python run_tdgl_with_different_gamma.py --gamma 1.5 --noise_strength 0.1 --save_hdf5
# 文件名: gamma_1.5_noise_strength_0.1.h5
```

## 输出文件说明

### HDF5 文件 (.h5)
- 包含完整的时间演化数据
- 可以使用 `tdgl.Solution.from_hdf5()` 加载
- 只在使用 `--save_hdf5` 时保存

### NPZ 文件 (.npz)
- 包含 I-V 数据和所有参数
- 总是保存
- 可以使用 `np.load()` 加载

## 注意事项

1. **文件名长度**: 如果修改很多参数，文件名会很长
2. **覆盖警告**: tdgl 会自动为已存在的文件名添加后缀，避免覆盖
3. **参数记录**: NPZ 文件中保存了所有参数值，方便后续分析
4. **噪声**: 噪声强度为 0 时不添加噪声（确定性模拟）

## 批量运行示例

```bash
# 扫描不同的 gamma 值
for gamma in 0.5 1.0 1.5 2.0; do
    python run_tdgl_with_different_gamma.py --gamma $gamma --save_hdf5
done

# 扫描不同的噪声强度
for noise in 0.0 0.05 0.1 0.2; do
    python run_tdgl_with_different_gamma.py --noise_strength $noise --save_hdf5
done
```
