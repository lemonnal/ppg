# PPG 信号处理与 SpO₂ 分析系统

一个基于 C++ 的脉搏血氧饱和度（SpO₂）信号处理和分析系统，采用高性能 DSP 滤波算法，支持离线批量处理和实时流式处理。

## 项目简介

本项目实现了完整的 PPG（光电容积脉搏波）信号处理流程，从原始信号预处理到血氧饱和度估算。系统采用 C++ 实现高性能数字信号处理，支持两种工作模式：
- **离线处理模式**：使用零相位滤波（filtfilt）对完整信号进行分析，适合后处理场景
- **实时处理模式**：使用单向 IIR 滤波器逐样本处理，适合嵌入式设备和实时监测场景

### 核心功能

#### 信号处理
- **零相位滤波**：实现 Python `scipy.signal.filtfilt` 功能，消除相位失真，适合离线分析
- **实时 IIR 滤波**：单向滤波支持逐样本处理，低延迟，适合实时系统
- **PPG 专用滤波器**：Butterworth 带通滤波（0.5-20Hz），有效去除基线漂移和高频噪声
- **滤波器预热**：支持均值初始化，减少滤波器瞬态响应

#### 分析算法
- **峰值检测**：仿照 `scipy.signal.find_peaks` 实现，支持距离、高度、显著性约束
- **谷值检测**：自动识别 PPG 信号谷值
- **心率计算**：基于峰值间隔计算 BPM 和 HRV（心率变异性）
- **SpO₂ 估算**：基于 AC/DC 比率法计算血氧饱和度

#### 工程特性
- **模块化设计**：清晰的头文件/源文件分离，易于维护和扩展
- **批量处理**：支持通过配置文件批量处理多个数据文件
- **实时缓冲**：滑动窗口缓冲区实现实时数据流处理

## 目录结构

```
workspace-ppg/
├── CMakeLists.txt               # CMake 构建配置
├── record.txt                   # 离线处理文件列表配置
│
├── include/                     # 头文件目录
│   ├── signal_io.hpp            # 信号输入输出接口
│   ├── ppg_filters.hpp          # PPG 滤波器（零相位+单向）
│   ├── ppg_analysis.hpp         # PPG 分析算法（峰值、心率、SpO₂）
│   ├── signal_utils.hpp         # 信号工具函数
│   ├── find_peaks.hpp           # 峰值检测（仿 scipy）
│   └── realtime_filter.hpp      # 实时滤波器和缓冲区
│
├── src/                         # 源文件目录
│   ├── signal_io.cpp            # 信号 I/O 实现
│   ├── ppg_filters.cpp          # 滤波器实现
│   ├── ppg_analysis.cpp         # 分析算法实现
│   ├── signal_utils.cpp         # 工具函数实现
│   ├── find_peaks.cpp           # 峰值检测实现
│   └── realtime_filter.cpp      # 实时滤波器实现
│
├── offline_main.cpp             # 离线处理程序入口
├── realtime_main.cpp            # 实时处理程序入口
│
├── build_and_run_offline.sh     # 离线处理构建+运行脚本
├── build_and_run_realtime.sh    # 实时处理构建+运行脚本
│
├── DSPFilters/                  # 第三方 DSP 滤波器库
│   ├── include/                 # 库头文件
│   ├── source/                  # 库源文件
│   └── common.mk                # 构建配置
│
├── DataSet/                     # 数据集目录
│   ├── PPG-BP/                  # PPG-BP 数据集（1000Hz）
│   └── RW-PPG/                  # RW-PPG 数据集
│
├── output_data/                 # 滤波输出数据目录
│
├── aaaInfo/                     # 技术文档
│   ├── 核心物理原理：朗伯-比尔定律.md
│   ├── DSPFilters库调用分析.md
│   └── 零相位滤波详解.md
│
├── aaaPyTest/                   # Python 辅助模块
│   ├── Method.py                # SpO₂ 计算核心算法（AC/DC比率法）
│   ├── concat_dataset.py        # 数据集拼接工具（循环扩展数据）
│   ├── show.py                  # 信号可视化对比工具（C++ vs Python）
│   ├── rw-ppg/                  # RW-PPG 数据集处理
│   │   └── rw_ppg.py            # RW-PPG 数据集SpO₂计算与分析
│   ├── but-ppg/                 # BUT-PPG 数据集处理
│   │   └── but_ppg.py           # BUT-PPG 数据集读取与解析
│   └── ppg-bp/                  # PPG-BP 数据集处理
│       └── ppg-bp.py            # PPG-BP 数据集SpO₂批量计算
│
├── BUILD_GUIDE.md               # 构建指南
└── README.md                    # 本文件
```

## 技术原理

### 算法基础

本项目基于**朗伯-比尔定律**和**AC/DC 比率法**实现无创血氧饱和度估算：

1. **信号分离**：将 PPG 信号分解为交流分量（AC）和直流分量（DC）
2. **峰值检测**：识别脉搏搏动引起的信号变化
3. **比率计算**：计算 AC/DC 比率，消除个体差异
4. **SpO₂ 估算**：使用经验公式将比率转换为血氧饱和度百分比

### 滤波技术

#### 零相位滤波（filtfilt）
- **原理**：正向滤波 → 反转信号 → 反向滤波 → 再次反转
- **优点**：完全消除相位失真，零群延迟
- **缺点**：需要完整信号，无法实时处理
- **适用场景**：离线数据分析、科研研究

#### 实时 IIR 滤波
- **原理**：单向通过 Butterworth IIR 滤波器
- **优点**：低延迟，逐样本处理，内存占用小
- **缺点**：存在相位失真（群延迟）
- **适用场景**：嵌入式设备、实时监测

#### 带通滤波器参数
- **类型**：Butterworth 带通滤波器
- **阶数**：3 阶（可配置）
- **通带范围**：0.5 - 20 Hz
  - 低截止 0.5Hz：去除基线漂移和运动伪影
  - 高截止 20Hz：去除高频噪声，保留心率相关频率

## 环境要求

### 编译环境

- **编译器**：支持 C++11 的编译器（GCC 4.8+, Clang 3.3+, MSVC 2015+）
- **构建工具**：CMake 3.10+
- **操作系统**：Linux（推荐）、macOS、Windows
- **CPU 核心**：支持多核并行编译

### 第三方依赖

- **DSPFilters**：C++ 滤波器库（已包含在项目中）
  - 提供多种 IIR/FIR 滤波器实现
  - 支持 Butterworth、Chebyshev、Elliptic 等类型

## 快速开始

### 一、离线处理模式

适用于已保存的 PPG 数据文件批量处理：

```bash
# 使用脚本自动构建并运行
./build_and_run_offline.sh

# 或者仅构建不运行
./build_and_run_offline.sh -n

# Debug 模式构建
./build_and_run_offline.sh -d
```

### 二、实时处理模式

适用于模拟实时信号流处理：

```bash
# 使用脚本自动构建并运行
./build_and_run_realtime.sh

# 或者仅构建不运行
./build_and_run_realtime.sh -n
```

### 三、手动构建

```bash
# 创建构建目录
mkdir build && cd build

# CMake 配置（Release 模式）
cmake -DCMAKE_BUILD_TYPE=Release ..

# 编译
make -j$(nproc)

# 运行离线处理
./offline_main

# 运行实时处理
./realtime_main
```

## 使用说明

### 离线处理模式

#### 配置输入文件

在项目根目录的 [record.txt](record.txt) 中配置要处理的文件列表：

```
259_3
2_1
2_2
3_1
```

每行一个文件名（不含扩展名），程序会自动从 `DataSet/PPG-BP/` 目录读取对应的 `.txt` 文件。

#### 程序参数配置

编辑 [offline_main.cpp](offline_main.cpp) 修改处理参数：

```cpp
// 滤波器参数
const double low_freq = 0.5;     // 低截止频率 (Hz)
const double high_freq = 20.0;   // 高截止频率 (Hz)
const double sample_rate = 1000.0; // 采样率 (Hz)
const int filter_order = 3;      // 滤波器阶数

// 滤波模式选择
int filter_method = 1;  // 1: 零相位滤波, 2: 单向IIR滤波

// 读取样本数
const int max_samples = 2100;  // 读取前2100个样本
```

#### 输出结果

滤波后的信号保存在 `output_data/` 目录：

| 文件名格式 | 说明 |
|-----------|------|
| `<filename>_filtered_zerophase.txt` | 零相位滤波结果 |
| `<filename>_filtered_oneway.txt` | 单向 IIR 滤波结果 |

### 实时处理模式

实时模式模拟嵌入式设备环境，逐样本处理数据流：

#### 配置参数

编辑 [realtime_main.cpp](realtime_main.cpp) 修改实时处理参数：

```cpp
// 数据源配置
const std::string data_file = "path/to/your/data.txt";
const double SAMPLE_RATE = 1000.0;  // 采样率 (Hz)

// 滤波器配置
const double LOW_FREQ = 0.5;        // 低频截止
const double HIGH_FREQ = 20.0;      // 高频截止
const int FILTER_ORDER = 3;         // 滤波器阶数

// 缓冲区配置
const size_t BUFFER_SIZE = 3000;     // 3秒数据 @ 1000Hz
const size_t ANALYSIS_WINDOW = 2100; // 分析窗口大小
const size_t UPDATE_INTERVAL = 1200;  // 分析更新间隔

// 实时模拟配置
const bool SIMULATE_REALTIME = false;  // 是否添加真实延迟
```

#### 输出信息

实时处理过程中会显示：
- 当前处理进度（样本数、时间）
- 检测到的峰值和谷值数量
- 实时心率（BPM）和 HRV
- 实时 SpO₂ 估算值

## API 参考

### 滤波器 API

#### 零相位滤波

```cpp
#include "ppg_filters.hpp"

// 应用零相位带通滤波
std::vector<float> filtered = ppg::apply_bandpass_zerophase(
    input_signal,    // 输入信号
    0.5,             // 低截止频率
    20.0,            // 高截止频率
    1000.0,          // 采样率
    3                // 滤波器阶数
);
```

#### 单向 IIR 滤波

```cpp
// 应用单向带通滤波
std::vector<float> filtered = ppg::apply_bandpass_oneway(
    input_signal,    // 输入信号
    0.5,             // 低截止频率
    20.0,            // 高截止频率
    1000.0,          // 采样率
    3,               // 滤波器阶数
    true             // 是否预热
);
```

#### 实时滤波器

```cpp
#include "realtime_filter.hpp"

// 创建实时滤波器
ppg::RealtimeFilter filter(0.5, 20.0, 1000.0, 3);

// 预热滤波器
filter.warmup(initial_value, 100);

// 逐样本处理
while (has_data) {
    float filtered = filter.process_sample(raw_sample);
    // 处理 filtered...
}
```

### 分析算法 API

```cpp
#include "ppg_analysis.hpp"

// 峰值和谷值检测
std::vector<int> peaks, valleys;
float ac_component;
ppg::detect_peaks_and_valleys(
    filtered_signal, 1000.0, 0.4,
    peaks, valleys, ac_component
);

// 计算心率
float heart_rate, hrv;
bool hr_valid = ppg::calculate_heart_rate(
    peaks, 1000.0, heart_rate, hrv
);

// 计算 SpO2
float spo2, ratio;
bool spo2_valid = ppg::calculate_spo2_from_ppg(
    input_signal, filtered_signal,
    peaks, valleys, ac_component,
    spo2, ratio
);
```

### 峰值检测 API

```cpp
#include "find_peaks.hpp"

// 基础峰值检测
std::vector<int> peaks = find_peaks(
    signal,          // 输入信号
    40,              // 最小间距（样本数）
    0.0f,            // 最小高度
    -1.0f            // 最小显著性（-1表示禁用）
);
```

## 数据集支持

### PPG-BP 数据集

- **采样率**：1000 Hz
- **信号类型**：单通道 PPG
- **文件格式**：纯文本，每行一个样本值
- **默认读取**：前 2100 个样本（2.1秒）
- **用途**：血压相关研究、心率变异性分析

### RW-PPG 数据集

- **采样率**：可变（25-100 Hz）
- **信号类型**：多通道 PPG（红光、红外）
- **用途**：可穿戴设备算法优化

## 性能特点

| 特性 | 离线模式 | 实时模式 |
|------|---------|---------|
| 滤波方法 | 零相位（filtfilt） | 单向 IIR |
| 相位失真 | 无 | 有（群延迟） |
| 延迟 | 高（需完整信号） | 低（逐样本） |
| 内存占用 | 中等（2x 缓冲） | 小（固定缓冲区） |
| 适用场景 | 数据分析、科研 | 嵌入式、实时监测 |

### 性能指标

- **处理速度**：支持 1000Hz 采样率实时处理
- **延迟**：实时模式下单样本滤波延迟 < 1μs
- **内存占用**：固定缓冲区模式下内存占用可控
- **精度**：单精度浮点运算，满足医疗级精度要求

## 命令行脚本参数

### build_and_run_offline.sh

| 参数 | 说明 |
|------|------|
| `-d, --debug` | 使用 Debug 模式构建 |
| `-n, --no-run` | 只构建不运行 |
| `-h, --help` | 显示帮助信息 |

### build_and_run_realtime.sh

| 参数 | 说明 |
|------|------|
| `-d, --debug` | 使用 Debug 模式构建 |
| `-n, --no-run` | 只构建不运行 |
| `-h, --help` | 显示帮助信息 |

## 技术文档

详细技术文档请参考 [aaaInfo/](aaaInfo/) 目录：

- **[核心物理原理：朗伯-比尔定律.md](aaaInfo/核心物理原理：朗伯-比尔定律.md)**：SpO₂ 测量的物理基础
- **[DSPFilters库调用分析.md](aaaInfo/DSPFilters库调用分析.md)**：滤波器库使用说明
- **[零相位滤波详解.md](aaaInfo/零相位滤波详解.md)**：零相位滤波实现原理

### 构建指南

详细的构建说明请参考 [BUILD_GUIDE.md](BUILD_GUIDE.md)

## 应用场景

- **可穿戴设备**：智能手表、手环的血氧监测
- **医疗设备**：便携式血氧仪的信号处理
- **健康监测**：家庭健康监护系统
- **运动科学**：运动过程中的血氧变化监测
- **睡眠分析**：睡眠呼吸暂停综合征筛查
- **嵌入式系统**：资源受限环境的实时信号处理

## 代码结构说明

### 头文件（include/）

| 文件 | 功能 |
|------|------|
| [signal_io.hpp](include/signal_io.hpp) | 信号文件读写接口 |
| [ppg_filters.hpp](include/ppg_filters.hpp) | 零相位和单向滤波器 |
| [ppg_analysis.hpp](include/ppg_analysis.hpp) | 峰值检测、心率、SpO₂计算 |
| [find_peaks.hpp](include/find_peaks.hpp) | scipy兼容的峰值检测 |
| [signal_utils.hpp](include/signal_utils.hpp) | 信号统计和工具函数 |
| [realtime_filter.hpp](include/realtime_filter.hpp) | 实时滤波器和滑动窗口缓冲区 |

### 源文件（src/）

对应的 `.cpp` 实现文件，包含所有算法的具体实现。

### 主程序

| 文件 | 功能 |
|------|------|
| [offline_main.cpp](offline_main.cpp) | 离线批量处理入口 |
| [realtime_main.cpp](realtime_main.cpp) | 实时流式处理入口 |

## 许可证

本项目使用第三方 DSPFilters 库，请遵守相应的许可证要求。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。

## Python 辅助模块说明

本项目提供了一套 Python 辅助工具模块，用于数据分析、算法验证和结果可视化。这些模块位于 [aaaPyTest/](aaaPyTest/) 目录下。

### 核心模块

#### Method.py - SpO₂ 计算核心算法

实现基于 AC/DC 比率法的 SpO₂ 估算算法，与 C++ 实现保持一致。

**核心功能**：
- 零相位高通滤波（0.5Hz）去除基线漂移
- 峰值和谷值检测（基于 scipy.signal.find_peaks）
- AC/DC 比率计算
- 三次多项式经验公式转换 AC/DC 比率为 SpO₂ 值

**使用示例**：
```python
from Method import calculate_spo2_from_ppg

spo2, ratio = calculate_spo2_from_ppg(
    ppg_signal,        # PPG信号数组
    sampling_rate=50,  # 采样率 (Hz)
    time_interval=0.4  # 峰值检测间隔 (秒)
)
print(f"SpO2: {spo2:.2f}%, AC/DC Ratio: {ratio:.4f}")
```

**经验公式**：
```python
spo2 = (-3.7465271198e+01 * ratio**3 +
         5.8403912586e+01 * ratio**2 +
        -3.7079378855e+01 * ratio +
         1.0016136403e+02)
spo2 = clip(spo2, 90, 100)  # 限制在 90-100% 范围
```

### 数据集处理模块

#### rw-ppg/rw_ppg.py - RW-PPG 数据集处理

处理 RW-PPG（Reflectance Wearable PPG）数据集，计算并分析 SpO₂ 值。

**数据集格式**：
- 训练集：1374 条信号，300 个采样点，50 Hz
- 测试集：700 条信号，300 个采样点，50 Hz
- 文件格式：Excel（.xlsx）

**主要功能**：
- 读取训练集和测试集 Excel 文件
- 信号可视化（8 通道子图展示）
- 批量计算所有信号的 SpO₂ 值
- 生成统计分析报告
- 输出到 Excel 文件（包含 Training_Set、Test_Set、Statistics 三个 sheet）
- 绘制 SpO₂ 分布图和箱线图

**输出文件**：
- `rw-ppg/rw_ppg_signals.png` - 信号样本图
- `rw-ppg/rw_ppg_spo2_data.xlsx` - SpO₂ 数据
- `rw-ppg/rw_ppg_spo2_analysis.png` - SpO₂ 分析图

#### ppg-bp/ppg-bp.py - PPG-BP 数据集处理

批量处理 PPG-BP 数据集中的所有 PPG 信号文件。

**数据集格式**：
- 采样率：1000 Hz
- 文件格式：纯文本（.txt），每行一条信号
- 自动扫描目录下所有 txt 文件

**主要功能**：
- 自动扫描并排序目录中的所有 txt 文件
- 按行读取每条信号（制表符分隔）
- 批量计算 SpO₂ 值和 AC/DC 比率
- 生成完整的统计报告
- 绘制信号样本和 SpO₂ 分析图

**输出文件**：
- `ppg-bp/ppg_bp_spo2_data.xlsx` - SpO₂ 批量计算结果

#### but-ppg/but_ppg.py - BUT-PPG 数据集读取

读取 BUT-PPG 数据集的 WFDB 格式文件。

**数据集格式**：
- 文件格式：WFDB 格式（.dat + .hea 头文件）
- 命名规则：`<record_name>_PPG.dat` / `<record_name>_PPG.hea`

**主要功能**：
- 自动扫描并提取所有记录名
- 使用 `wfdb.rdrecord()` 读取信号数据
- 使用 `wfdb.rdheader()` 读取头信息
- 显示数据集结构信息

### 工具模块

#### concat_dataset.py - 数据集拼接工具

将短 PPG 信号循环拼接，用于生成更长的测试数据。

**主要功能**：
- 读取指定 PPG 文件
- 将原始数据循环拼接 N 次（默认 8 次）
- 保存拼接后的数据到新文件
- 绘制拼接后的信号波形

**使用场景**：
- 将短时信号扩展为长时间测试数据
- 验证实时滤波器的稳定性
- 测试算法在长时间数据上的表现

**输出文件**：
- `concat_<record_name>.txt` - 拼接后的数据
- `concat_<record_name>.png` - 波形图

#### show.py - 信号可视化对比

对比 C++ 和 Python 滤波结果的差异，用于算法验证。

**主要功能**：
- 读取 C++ 滤波输出文件
- 读取 Python 滤波输出文件
- 绘制两者波形对比图
- 计算并显示误差统计信息
- 验证 C++ 实现的正确性

### Python 环境依赖

```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
pandas>=1.3.0
openpyxl>=3.0.0
wfdb>=3.4.0
```

安装依赖：
```bash
pip install numpy scipy matplotlib pandas openpyxl wfdb
```

### 使用示例

**处理 RW-PPG 数据集**：
```bash
cd aaaPyTest/rw-ppg
python rw_ppg.py
```

**处理 PPG-BP 数据集**：
```bash
cd aaaPyTest/ppg-bp
python ppg-bp.py
```

**拼接数据集**：
```bash
cd aaaPyTest
# 编辑 concat_dataset.py 修改 record_name 和拼接倍数
python concat_dataset.py
```