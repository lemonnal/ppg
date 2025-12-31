# PPG 信号处理与 SpO₂ 分析系统

一个基于 C++ 和 Python 的脉搏血氧饱和度（SpO₂）信号处理和分析系统，结合高性能 DSP 滤波与灵活的数据分析能力。

## 项目简介

本项目实现了完整的 PPG（光电容积脉搏波）信号处理流程，从原始信号预处理到血氧饱和度估算。系统采用 C++ 实现高性能数字信号处理，使用 Python 进行数据分析和可视化，适用于可穿戴设备、医疗监测和健康管理等应用场景。

### 核心功能

- **零相位滤波**：实现 Python `filtfilt` 功能，消除相位失真
- **PPG 专用滤波器**：Butterworth 带通滤波（0.5-20Hz）、基线漂移去除
- **峰值检测**：自动识别 PPG 信号峰值和谷值
- **SpO₂ 估算**：基于 AC/DC 比率法计算血氧饱和度
- **多数据集支持**：支持 BUT-PPG、RW-PPG、PPG-BP 等数据集
- **批量处理**：自动化处理大规模 PPG 数据
- **数据可视化**：信号波形和 SpO₂ 分布图表

## 目录结构

```
workspace-ppg/
├── DSPFilter/                    # C++ 核心 DSP 库
│   ├── include/                  # 头文件
│   │   ├── signal_io.hpp         # 信号输入输出
│   │   ├── ppg_filters.hpp       # PPG 滤波器
│   │   ├── ppg_analysis.hpp      # PPG 分析算法
│   │   ├── signal_utils.hpp      # 信号工具函数
│   │   └── find_peaks.hpp        # 峰值检测
│   ├── src/                      # 源文件
│   ├── build/                    # CMake 构建目录
│   ├── output_data/              # 滤波输出数据
│   ├── CMakeLists.txt            # CMake 配置
│   └── BUILD_GUIDE.md            # 构建指南
├── python_test/                  # Python 实现
│   ├── Method.py                 # SpO₂ 计算核心算法
│   ├── but-ppg/                  # BUT-PPG 数据集处理
│   └── rw-ppg/                   # RW-PPG 数据集处理
├── DataSet/                      # 数据集目录
│   ├── BUT-PPG/                  # 100Hz PPG 数据集
│   ├── PPG-BP/                   # 血压相关 PPG 数据集
│   └── RW-PPG/                   # 可穿戴设备 PPG 数据集
└── information/                  # 技术文档
    ├── 核心物理原理：朗伯-比尔定律.md
    ├── DSPFilters库调用分析.md
    └── 零相位滤波详解.md
```

## 技术原理

### 算法基础

本项目基于**朗伯-比尔定律**和**AC/DC 比率法**实现无创血氧饱和度估算：

1. **信号分离**：将 PPG 信号分解为交流分量（AC）和直流分量（DC）
2. **峰值检测**：识别脉搏搏动引起的信号变化
3. **比率计算**：计算 AC/DC 比率，消除个体差异
4. **SpO₂ 估算**：使用经验公式将比率转换为血氧饱和度百分比

### 滤波技术

- **零相位滤波**：双向滤波消除相位失真，适合离线分析
- **实时 IIR 滤波**：单向滤波支持实时信号处理
- **带通滤波**：0.5-20Hz Butterworth 滤波器保留有效频率成分

## 环境要求

### C++ 环境

- **编译器**：支持 C++11 的编译器（GCC 4.8+, Clang 3.3+, MSVC 2015+）
- **构建工具**：CMake 3.10+
- **操作系统**：Linux, macOS, Windows

### Python 环境

- **Python 版本**：Python 3.6+
- **核心依赖**：
  ```bash
  pip install numpy scipy pandas matplotlib wfdb openpyxl scikit-learn
  ```

## 快速开始

### C++ 模块构建

1. 进入 C++ 项目目录：
```bash
cd DSPFilter
```

2. 创建构建目录并配置：
```bash
mkdir -p build && cd build
cmake ..
```

3. 编译项目：
```bash
make
```

或使用提供的构建脚本：
```bash
bash cmake_build.sh
```

4. 运行程序：
```bash
./DSPFilter
```

### Python 模块使用

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 处理 RW-PPG 数据集：
```bash
cd python_test/rw-ppg
python rw_ppg.py
```

3. 处理 BUT-PPG 数据集：
```bash
cd python_test/but-ppg
python but_ppg.py
```

## 使用说明

### C++ 模块

#### 输入数据准备

在 `DSPFilter/record.txt` 中配置输入文件列表：
```
data_path/sample_1.txt
data_path/sample_2.txt
```

#### 滤波器配置

在 `main.cpp` 中配置滤波器参数：
```cpp
// Butterworth 带通滤波器
double low_cutoff = 0.5;   // 低截止频率
double high_cutoff = 20.0; // 高截止频率
int filter_order = 4;      // 滤波器阶数
```

#### 输出结果

滤波后的信号保存在 `output_data/` 目录，文件命名格式：
- `<filename>_filtered.txt`：滤波后信号
- `<filename>_peaks.txt`：峰值位置

### Python 模块

#### SpO₂ 计算

使用 `Method.py` 中的核心算法：
```python
from Method import calculate_spo2

spo2 = calculate_spo2(red_signal, ir_signal)
print(f"SpO₂: {spo2:.2f}%")
```

#### 数据可视化

生成信号波形图和 SpO₂ 分布图：
```python
import matplotlib.pyplot as plt

# 绘制 PPG 信号
plt.plot(time, ppg_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
```

## 数据集支持

### BUT-PPG 数据集

- **采样率**：100 Hz
- **信号类型**：单通道 PPG
- **用途**：基础算法验证

### RW-PPG 数据集

- **采样率**：可变（25-100 Hz）
- **信号类型**：多通道 PPG（红光、红外）
- **用途**：可穿戴设备算法优化

### PPG-BP 数据集

- **采样率**：100 Hz
- **信号类型**：PPG + 血压标签
- **用途**：血压估算研究

## 性能特点

- **高性能**：C++ 实现的 DSP 滤波，适合实时处理
- **零延迟**：零相位滤波消除群延迟
- **高精度**：双精度浮点运算，确保计算精度
- **可扩展**：模块化设计，易于添加新的滤波器和分析算法

## 技术文档

详细技术文档请参考 [information/](information/) 目录：

- **[核心物理原理：朗伯-比尔定律.md](information/核心物理原理：朗伯-比尔定律.md)**：SpO₂ 测量的物理基础
- **[DSPFilters库调用分析.md](information/DSPFilters库调用分析.md)**：滤波器库使用说明
- **[零相位滤波详解.md](information/零相位滤波详解.md)**：零相位滤波实现原理

## 构建指南

详细的构建说明请参考 [DSPFilter/BUILD_GUIDE.md](DSPFilter/BUILD_GUIDE.md)

## 应用场景

- **可穿戴设备**：智能手表、手环的血氧监测
- **医疗设备**：便携式血氧仪的信号处理
- **健康监测**：家庭健康监护系统
- **运动科学**：运动过程中的血氧变化监测
- **睡眠分析**：睡眠呼吸暂停综合征筛查

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。

---

**注意**：本系统仅用于研究和教育目的，不应直接用于医疗诊断。临床应用需经过严格的医疗器械认证。
