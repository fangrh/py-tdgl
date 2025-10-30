# run_tdgl_with_different_gamma.py 参数说明

## 命令行参数列表

### 物理参数
- `--gamma`: Gamma 参数 (default: 1.0)
- `--u`: u 参数 (default: 5.79)
- `--d`: 薄膜厚度 (default: 0.01)
- `--xi`: 相干长度 (default: 0.25)
- `--london_lambda`: London 穿透深度 (default: 0.25)
- `--sigma`: 正常态电导率 (default: 1000)

### 热扩散参数
- `--use_heat`: 是否启用热扩散 (default: True)
- `--T_0`: 无量纲临界温度 (default: 1.0)
- `--kappa_eff`: 有效热导率（无量纲）(default: 0.02)
- `--C_eff`: 有效热容（无量纲）(default: 1.0)
- `--T_heat`: 环境温度（无量纲）(default: 0.1)
- `--hole_eta`: hole 区域的热交换系数 (default: 0.05)
- `--environment_eta`: 外部区域的热交换系数 (default: 10.0)
- `--hole_gap`: hole 区域的半宽度（μm）(default: 1.0)

### 几何参数
- `--width`: 薄膜宽度（μm）(default: 6.0)
- `--length`: 薄膜长度（μm）(default: 18.0)
- `--electride_distance`: 电极之间的距离（μm）(default: 8.0)
- `--probe_distance`: 探测点距离中心的距离（μm）(default: 2.5)

### 时间参数 ⭐ 新增
- `--ramp_up_time`: 电流上升时间 (default: 5000)
- `--max_current_time`: 保持最大电流的时间 (default: 0)
- `--ramp_down_time`: 电流下降时间 (default: 5000)
- `--zero_current_time`: 保持零电流的时间 (default: 0)

**总模拟时间**: `solve_time = ramp_up_time + max_current_time + ramp_down_time + zero_current_time`

## 使用示例

### 示例 1: 基本运行（使用默认参数）
```bash
python run_tdgl_with_different_gamma.py
```

### 示例 2: 修改 gamma 和热参数
```bash
python run_tdgl_with_different_gamma.py \
    --gamma 2.0 \
    --hole_eta 0.1 \
    --environment_eta 20.0 \
    --kappa_eff 0.05
```

### 示例 3: 修改时间参数（快速测试）
```bash
python run_tdgl_with_different_gamma.py \
    --ramp_up_time 1000 \
    --ramp_down_time 1000
# 总时间 = 1000 + 0 + 1000 + 0 = 2000
```

### 示例 4: 更长的模拟时间
```bash
python run_tdgl_with_different_gamma.py \
    --ramp_up_time 10000 \
    --max_current_time 5000 \
    --ramp_down_time 10000 \
    --zero_current_time 2000
# 总时间 = 10000 + 5000 + 10000 + 2000 = 27000
```

### 示例 5: 完整参数组合
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

## 电流曲线说明

电流随时间的变化分为四个阶段：

```
电流 (I)
   │
Imax│    ┌──────────┐ Stage 2: max_current_time
   │   ╱│          │╲
   │  ╱ │          │ ╲
   │ ╱  │          │  ╲
  0├────┼──────────┼───┼─────── 时间 (t)
      Stage 1    Stage 3  Stage 4
   ramp_up_time  ramp_down zero_current
                   _time      _time
```

**Stage 1**: 从 0 线性增加到 Imax（持续 `ramp_up_time`）
**Stage 2**: 保持在 Imax（持续 `max_current_time`）
**Stage 3**: 从 Imax 线性减少到 0（持续 `ramp_down_time`）
**Stage 4**: 保持在 0（持续 `zero_current_time`）

## 输出文件命名

输出文件名包含所有重要参数：
```
gamma{gamma}_holeeta{hole_eta}_environmenteta{environment_eta}_...npz
```

保存的数据包括：
- `time`: 时间数组
- `currents`: 电流数组
- `voltages`: 电压差数组
- 所有输入参数（包括时间参数）

## 注意事项

1. **时间单位**：所有时间参数都是无量纲的（归一化后的时间）
2. **热扩散稳定性**：
   - 太小的 `ramp_up_time` 可能导致数值不稳定
   - 建议 `ramp_up_time >= 1000` 以确保稳定性
3. **模拟时长**：
   - 总时间过长会增加计算成本
   - 建议先用较短时间测试，确认无误后再延长
4. **hole_gap 单位**：注意 `hole_gap` 的单位是 **μm**（物理单位），会在代码中自动转换为无量纲单位

## 性能优化建议

- **快速测试**: `--ramp_up_time 500 --ramp_down_time 500`
- **标准模拟**: `--ramp_up_time 5000 --ramp_down_time 5000` (default)
- **详细研究**: `--ramp_up_time 10000 --max_current_time 2000 --ramp_down_time 10000`

## 批处理脚本示例

```bash
#!/bin/bash
# 扫描不同的 gamma 值和时间参数

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
