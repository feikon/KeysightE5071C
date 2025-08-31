# Keysight E5071C VNA 驱动程序使用指南

这是一个用于控制 Keysight E5071C 矢量网络分析仪的 Python 驱动程序。

## 快速开始

### 基本连接
```python
from keysight_E5071C import E5071C

# 连接到VNA (替换为实际IP地址)
vna = E5071C("TCPIP::192.168.1.100::INSTR")
print("设备信息:", vna.identify())
```

### 频率设置
```python
# 设置频率范围
vna.freq_start(1e9)    # 起始频率 1GHz
vna.freq_stop(2e9)     # 终止频率 2GHz
vna.points(1001)       # 测量点数
vna.bandwidth(1000)    # 中频带宽 1kHz

# 或者使用便捷方法
vna.set_freq_axis(start=1e9, stop=2e9, point=1001, bandwidth=1000)
```

### 迹线配置
```python
# 设置迹线数量
vna.traces_number(2)

# 配置第1条迹线
vna.active_trace(1)
vna.s_par('S21')           # 设置S21参数
vna.format_trace('mlog')   # 对数幅度显示

# 配置第2条迹线
vna.active_trace(2)
vna.s_par('S21')           # 设置S21参数
vna.format_trace('phase')  # 相位显示
```

### 触发和测量
```python
# 设置触发
vna.trigger_source('bus')      # 总线触发
vna.trigger_initiate('single') # 单次触发模式

# 执行测量
vna.trigger_now()

# 读取数据
freq = vna.read_freq()                    # 频率轴
real, imag = vna.read_trace(1)           # 第1条迹线
all_data = vna.read_all_traces()         # 所有迹线
```

## 主要功能模块

### 1. 频率轴设置
- `freq_start()`, `freq_stop()` - 起始/终止频率
- `freq_center()`, `freq_span()` - 中心频率/频率范围
- `points()` - 测量点数
- `bandwidth()` - 中频带宽
- `sweep_type()` - 扫描类型

### 2. 迹线管理
- `traces_number()` - 迹线数量
- `active_trace()` - 选择活动迹线
- `format_trace()` - 显示格式
- `s_par()` - S参数类型

### 3. 平均设置
- `average_state()` - 平均开关
- `average_count()` - 平均次数
- `average_reset()` - 重置平均

### 4. 触发控制
- `trigger_source()` - 触发源
- `trigger_initiate()` - 触发模式
- `trigger_now()` - 立即触发

### 5. 数据读取
- `read_freq()` - 频率轴数据
- `read_trace()` - 单条迹线数据
- `read_all_traces()` - 所有迹线数据

### 6. 输出设置
- `power()` - 输出功率
- `output()` - RF输出开关
- `delay()` - 电气延迟
- `phase_offset()` - 相位偏移

## 完整测量示例

```python
import numpy as np
from keysight_E5071C import E5071C

# 连接设备
vna = E5071C("TCPIP::192.168.1.100::INSTR")

# 设置测量参数
vna.set_freq_axis(start=1e9, stop=10e9, point=1001)
vna.set_trigger('bus', 0, 'single')
vna.output('on')
vna.power(-10)  # -10 dBm

# 配置S21测量
vna.traces_number(1)
vna.active_trace(1)
vna.s_par('S21')
vna.format_trace('mlog')

# 执行测量
print("开始测量...")
vna.trigger_now()

# 读取数据
freq = vna.read_freq()
real, imag = vna.read_trace(1)

# 计算幅度和相位
magnitude_db = 20 * np.log10(np.sqrt(real**2 + imag**2))
phase_deg = np.arctan2(imag, real) * 180 / np.pi

print(f"频率范围: {freq[0]/1e9:.2f} - {freq[-1]/1e9:.2f} GHz")
print(f"幅度范围: {magnitude_db.min():.2f} - {magnitude_db.max():.2f} dB")

# 关闭连接
vna.close()
```

## 错误处理

```python
try:
    vna = E5071C("TCPIP::192.168.1.100::INSTR")
    # 测量代码...
except Exception as e:
    print(f"连接错误: {e}")
finally:
    if 'vna' in locals():
        vna.close()
```

## 注意事项

1. **IP地址**: 确保VNA的IP地址设置正确
2. **VISA驱动**: 需要安装PyVISA和相应的VISA驱动
3. **数据格式**: 复数数据包含实部和虚部
4. **触发模式**: 使用平均时需要设置 `trigger_averaging(True)`
5. **连接管理**: 使用完毕后调用 `close()` 关闭连接

## 支持的S参数
- S11, S12, S13, S14
- S21, S22, S23, S24
- S31, S32, S33, S34
- S41, S42, S43, S44

## 支持的显示格式
- `mlog` - 对数幅度 (dB)
- `phase` - 相位 (度)
- `lin_mag` - 线性幅度
- `real` - 实部
- `imag` - 虚部
- `uph` - 展开相位
- `pph` - 正相位
- `plin` - 极坐标线性
- `plog` - 极坐标对数
