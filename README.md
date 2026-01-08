# PPG Signal Processing and SpO2 Analysis System

A C++ based Photoplethysmography (PPG) signal processing and analysis system featuring high-performance DSP filtering algorithms, supporting both offline batch processing and real-time streaming.

## Project Overview

This project implements a complete PPG (Photoplethysmography) signal processing pipeline, from raw signal preprocessing to blood oxygen saturation estimation. The system uses C++ for high-performance digital signal processing and supports two operating modes:
- **Offline Processing Mode**: Uses zero-phase filtering (filtfilt) for complete signal analysis, suitable for post-processing scenarios
- **Real-time Processing Mode**: Uses one-way IIR filters for sample-by-sample processing, suitable for embedded devices and real-time monitoring

### Core Features

#### Signal Processing
- **Zero-phase Filtering**: Implements Python's `scipy.signal.filtfilt` functionality, eliminating phase distortion, ideal for offline analysis
- **Real-time IIR Filtering**: One-way filtering supports sample-by-sample processing with low latency, suitable for real-time systems
- **PPG-specific Filters**: Butterworth bandpass filtering (0.5-20Hz), effectively removes baseline drift and high-frequency noise
- **Filter Warm-up**: Supports mean initialization to reduce filter transient response

#### Analysis Algorithms
- **Peak Detection**: Implemented following `scipy.signal.find_peaks`, supporting distance, height, and prominence constraints
- **Valley Detection**: Automatic identification of PPG signal valleys
- **Heart Rate Calculation**: Calculates BPM and HRV (Heart Rate Variability) based on peak intervals
- **SpO2 Estimation**: Calculates blood oxygen saturation based on AC/DC ratio method

#### Engineering Features
- **Modular Design**: Clear header/source file separation, easy to maintain and extend
- **Batch Processing**: Supports batch processing of multiple data files via configuration files
- **Real-time Buffering**: Sliding window buffer for real-time data stream processing

## Directory Structure

```
workspace-ppg/
├── CMakeLists.txt               # CMake build configuration
├── record.txt                   # Offline processing file list configuration
│
├── include/                     # Header files directory
│   ├── signal_io.hpp            # Signal input/output interfaces
│   ├── ppg_filters.hpp          # PPG filters (zero-phase + one-way)
│   ├── ppg_analysis.hpp         # PPG analysis algorithms (peaks, HR, SpO2)
│   ├── signal_utils.hpp         # Signal utility functions
│   ├── find_peaks.hpp           # Peak detection (scipy-like)
│   └── realtime_filter.hpp      # Real-time filters and buffers
│
├── src/                         # Source files directory
│   ├── signal_io.cpp            # Signal I/O implementation
│   ├── ppg_filters.cpp          # Filter implementation
│   ├── ppg_analysis.cpp         # Analysis algorithm implementation
│   ├── signal_utils.cpp         # Utility function implementation
│   ├── find_peaks.cpp           # Peak detection implementation
│   └── realtime_filter.cpp      # Real-time filter implementation
│
├── offline_main.cpp             # Offline processing program entry
├── realtime_main.cpp            # Real-time processing program entry
│
├── build_and_run_offline.sh     # Offline processing build + run script
├── build_and_run_realtime.sh    # Real-time processing build + run script
│
├── DSPFilters/                  # Third-party DSP filter library
│   ├── include/                 # Library headers
│   ├── source/                  # Library sources
│   └── common.mk                # Build configuration
│
├── DataSet/                     # Dataset directory
│   ├── PPG-BP/                  # PPG-BP dataset (1000Hz)
│   └── RW-PPG/                  # RW-PPG dataset
│
├── output_data/                 # Filtered output data directory
│
├── aaaInfo/                     # Technical documentation
│   ├── 核心物理原理：朗伯-比尔定律.md
│   ├── DSPFilters库调用分析.md
│   └── 零相位滤波详解.md
│
├── aaaPyTest/                   # Python auxiliary modules
│   ├── Method.py                # SpO2 calculation core algorithm (AC/DC ratio method)
│   ├── concat_dataset.py        # Dataset concatenation tool (cyclic data extension)
│   ├── show.py                  # Signal visualization comparison tool (C++ vs Python)
│   ├── rw-ppg/                  # RW-PPG dataset processing
│   │   └── rw_ppg.py            # RW-PPG dataset SpO2 calculation and analysis
│   ├── but-ppg/                 # BUT-PPG dataset processing
│   │   └── but_ppg.py           # BUT-PPG dataset reading and parsing
│   └── ppg-bp/                  # PPG-BP dataset processing
│       └── ppg-bp.py            # PPG-BP dataset SpO2 batch calculation
│
├── BUILD_GUIDE.md               # Build guide
└── README.md                    # This file
```

## Technical Principles

### Algorithm Foundations

This project implements non-invasive blood oxygen saturation estimation based on **Beer-Lambert Law** and **AC/DC Ratio Method**:

1. **Signal Separation**: Decomposes PPG signal into AC component and DC component
2. **Peak Detection**: Identifies signal changes caused by pulse beats
3. **Ratio Calculation**: Calculates AC/DC ratio to eliminate individual differences
4. **SpO2 Estimation**: Uses empirical formula to convert ratio to blood oxygen saturation percentage

### Filtering Techniques

#### Zero-phase Filtering (filtfilt)
- **Principle**: Forward filtering → Reverse signal → Backward filtering → Reverse again
- **Advantages**: Completely eliminates phase distortion, zero group delay
- **Disadvantages**: Requires complete signal, not suitable for real-time processing
- **Use Cases**: Offline data analysis, scientific research

#### Real-time IIR Filtering
- **Principle**: One-way pass through Butterworth IIR filter
- **Advantages**: Low latency, sample-by-sample processing, small memory footprint
- **Disadvantages**: Has phase distortion (group delay)
- **Use Cases**: Embedded devices, real-time monitoring

#### Bandpass Filter Parameters
- **Type**: Butterworth bandpass filter
- **Order**: 3rd order (configurable)
- **Passband Range**: 0.5 - 20 Hz
  - Low cutoff 0.5Hz: Removes baseline drift and motion artifacts
  - High cutoff 20Hz: Removes high-frequency noise, preserves heart-rate related frequencies

## Environment Requirements

### Compilation Environment

- **Compiler**: C++11 compatible compiler (GCC 4.8+, Clang 3.3+, MSVC 2015+)
- **Build Tool**: CMake 3.10+
- **Operating System**: Linux (recommended), macOS, Windows
- **CPU Cores**: Supports multi-core parallel compilation

### Third-party Dependencies

- **DSPFilters**: C++ filter library (included in the project)
  - Provides various IIR/FIR filter implementations
  - Supports Butterworth, Chebyshev, Elliptic and other types

## Quick Start

### 1. Offline Processing Mode

For batch processing of saved PPG data files:

```bash
# Use script to automatically build and run
./build_and_run_offline.sh

# Or build only without running
./build_and_run_offline.sh -n

# Build in Debug mode
./build_and_run_offline.sh -d
```

### 2. Real-time Processing Mode

For simulating real-time signal stream processing:

```bash
# Use script to automatically build and run
./build_and_run_realtime.sh

# Or build only without running
./build_and_run_realtime.sh -n
```

### 3. Manual Build

```bash
# Create build directory
mkdir build && cd build

# CMake configuration (Release mode)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Compile
make -j$(nproc)

# Run offline processing
./offline_main

# Run real-time processing
./realtime_main
```

## Usage Instructions

### Offline Processing Mode

#### Configure Input Files

Configure the list of files to process in [record.txt](record.txt) in the project root:

```
259_3
2_1
2_2
197_3_1
```

One filename per line (without extension), the program will automatically read the corresponding `.txt` file from the `DataSet/PPG-BP/` directory.

#### Program Parameter Configuration

Edit [offline_main.cpp](offline_main.cpp) to modify processing parameters:

```cpp
// Filter parameters
const double low_freq = 0.5;     // Low cutoff frequency (Hz)
const double high_freq = 20.0;   // High cutoff frequency (Hz)
const double sample_rate = 1000.0; // Sample rate (Hz)
const int filter_order = 3;      // Filter order

// Filter mode selection
int filter_method = 1;  // 1: Zero-phase filtering, 2: One-way IIR filtering

// Number of samples to read
const int max_samples = 2100;  // Read first 2100 samples
```

#### Output Results

Filtered signals are saved in the `output_data/` directory:

| Filename Format | Description |
|----------------|-------------|
| `<filename>_filtered_zerophase.txt` | Zero-phase filtering result |
| `<filename>_filtered_oneway.txt` | One-way IIR filtering result |

### Real-time Processing Mode

Real-time mode simulates an embedded device environment, processing data streams sample-by-sample:

#### Configure Parameters

Edit [realtime_main.cpp](realtime_main.cpp) to modify real-time processing parameters:

```cpp
// Data source configuration
const std::string data_file = "path/to/your/data.txt";
const double SAMPLE_RATE = 1000.0;  // Sample rate (Hz)

// Filter configuration
const double LOW_FREQ = 0.5;        // Low frequency cutoff
const double HIGH_FREQ = 20.0;      // High frequency cutoff
const int FILTER_ORDER = 3;         // Filter order

// Buffer configuration
const size_t BUFFER_SIZE = 3000;     // 3 seconds of data @ 1000Hz
const size_t ANALYSIS_WINDOW = 2100; // Analysis window size
const size_t UPDATE_INTERVAL = 1200;  // Analysis update interval

// Real-time simulation configuration
const bool SIMULATE_REALTIME = false;  // Whether to add real delay
```

#### Output Information

During real-time processing, the following is displayed:
- Current processing progress (sample count, time)
- Number of detected peaks and valleys
- Real-time heart rate (BPM) and HRV
- Real-time SpO2 estimation value

## API Reference

### Filter API

#### Zero-phase Filtering

```cpp
#include "ppg_filters.hpp"

// Apply zero-phase bandpass filtering
std::vector<float> filtered = ppg::apply_bandpass_zerophase(
    input_signal,    // Input signal
    0.5,             // Low cutoff frequency
    20.0,            // High cutoff frequency
    1000.0,          // Sample rate
    3                // Filter order
);
```

#### One-way IIR Filtering

```cpp
// Apply one-way bandpass filtering
std::vector<float> filtered = ppg::apply_bandpass_oneway(
    input_signal,    // Input signal
    0.5,             // Low cutoff frequency
    20.0,            // High cutoff frequency
    1000.0,          // Sample rate
    3,               // Filter order
    true             // Whether to warm up
);
```

#### Real-time Filter

```cpp
#include "realtime_filter.hpp"

// Create real-time filter
ppg::RealtimeFilter filter(0.5, 20.0, 1000.0, 3);

// Warm up filter
filter.warmup(initial_value, 100);

// Process sample by sample
while (has_data) {
    float filtered = filter.process_sample(raw_sample);
    // Process filtered...
}
```

### Analysis Algorithm API

```cpp
#include "ppg_analysis.hpp"

// Peak and valley detection
std::vector<int> peaks, valleys;
float ac_component;
ppg::detect_peaks_and_valleys(
    filtered_signal, 1000.0, 0.4,
    peaks, valleys, ac_component
);

// Calculate heart rate
float heart_rate, hrv;
bool hr_valid = ppg::calculate_heart_rate(
    peaks, 1000.0, heart_rate, hrv
);

// Calculate SpO2
float spo2, ratio;
bool spo2_valid = ppg::calculate_spo2_from_ppg(
    input_signal, filtered_signal,
    peaks, valleys, ac_component,
    spo2, ratio
);
```

### Peak Detection API

```cpp
#include "find_peaks.hpp"

// Basic peak detection
std::vector<int> peaks = find_peaks(
    signal,          // Input signal
    40,              // Minimum spacing (sample count)
    0.0f,            // Minimum height
    -1.0f            // Minimum prominence (-1 means disabled)
);
```

## Dataset Support

### PPG-BP Dataset

- **Sample Rate**: 1000 Hz
- **Signal Type**: Single-channel PPG
- **File Format**: Plain text, one sample value per line
- **Default Read**: First 2100 samples (2.1 seconds)
- **Usage**: Blood pressure related research, heart rate variability analysis

### RW-PPG Dataset

- **Sample Rate**: Variable (25-100 Hz)
- **Signal Type**: Multi-channel PPG (red, infrared)
- **Usage**: Wearable device algorithm optimization

## Performance Characteristics

| Feature | Offline Mode | Real-time Mode |
|---------|--------------|----------------|
| Filter Method | Zero-phase (filtfilt) | One-way IIR |
| Phase Distortion | None | Yes (group delay) |
| Latency | High (requires complete signal) | Low (sample-by-sample) |
| Memory Usage | Medium (2x buffer) | Small (fixed buffer) |
| Use Cases | Data analysis, research | Embedded, real-time monitoring |

### Performance Metrics

- **Processing Speed**: Supports real-time processing at 1000Hz sample rate
- **Latency**: Single sample filtering delay < 1μs in real-time mode
- **Memory Usage**: Controllable memory usage in fixed buffer mode
- **Accuracy**: Single-precision floating-point arithmetic, meets medical-grade accuracy requirements

## Command Line Script Parameters

### build_and_run_offline.sh

| Parameter | Description |
|-----------|-------------|
| `-d, --debug` | Build in Debug mode |
| `-n, --no-run` | Build only without running |
| `-h, --help` | Display help information |

### build_and_run_realtime.sh

| Parameter | Description |
|-----------|-------------|
| `-d, --debug` | Build in Debug mode |
| `-n, --no-run` | Build only without running |
| `-h, --help` | Display help information |

## Technical Documentation

For detailed technical documentation, please refer to the [aaaInfo/](aaaInfo/) directory:

- **[核心物理原理：朗伯-比尔定律.md](aaaInfo/核心物理原理：朗伯-比尔定律.md)**: Physical foundation of SpO2 measurement
- **[DSPFilters库调用分析.md](aaaInfo/DSPFilters库调用分析.md)**: Filter library usage instructions
- **[零相位滤波详解.md](aaaInfo/零相位滤波详解.md)**: Zero-phase filtering implementation principles

### Build Guide

For detailed build instructions, please refer to [BUILD_GUIDE.md](BUILD_GUIDE.md)

## Application Scenarios

- **Wearable Devices**: Blood oxygen monitoring for smartwatches and fitness bands
- **Medical Equipment**: Signal processing for portable pulse oximeters
- **Health Monitoring**: Home health monitoring systems
- **Sports Science**: Blood oxygen change monitoring during exercise
- **Sleep Analysis**: Sleep apnea syndrome screening
- **Embedded Systems**: Real-time signal processing in resource-constrained environments

## Code Structure Description

### Header Files (include/)

| File | Function |
|------|----------|
| [signal_io.hpp](include/signal_io.hpp) | Signal file read/write interfaces |
| [ppg_filters.hpp](include/ppg_filters.hpp) | Zero-phase and one-way filters |
| [ppg_analysis.hpp](include/ppg_analysis.hpp) | Peak detection, heart rate, SpO2 calculation |
| [find_peaks.hpp](include/find_peaks.hpp) | Scipy-compatible peak detection |
| [signal_utils.hpp](include/signal_utils.hpp) | Signal statistics and utility functions |
| [realtime_filter.hpp](include/realtime_filter.hpp) | Real-time filters and sliding window buffers |

### Source Files (src/)

Corresponding `.cpp` implementation files containing the concrete implementation of all algorithms.

### Main Programs

| File | Function |
|------|----------|
| [offline_main.cpp](offline_main.cpp) | Offline batch processing entry |
| [realtime_main.cpp](realtime_main.cpp) | Real-time streaming processing entry |

## License

This project uses the third-party DSPFilters library. Please comply with the corresponding license requirements.

## Contributing

Issues and Pull Requests are welcome to improve this project.

## Python Auxiliary Modules Description

This project provides a set of Python auxiliary tool modules for data analysis, algorithm validation, and result visualization. These modules are located in the [aaaPyTest/](aaaPyTest/) directory.

### Core Modules

#### Method.py - SpO2 Calculation Core Algorithm

Implements SpO2 estimation algorithm based on AC/DC ratio method, consistent with the C++ implementation.

**Core Features**:
- Zero-phase high-pass filtering (0.5Hz) to remove baseline drift
- Peak and valley detection (based on scipy.signal.find_peaks)
- AC/DC ratio calculation
- Cubic polynomial empirical formula to convert AC/DC ratio to SpO2 value

**Usage Example**:
```python
from Method import calculate_spo2_from_ppg

spo2, ratio = calculate_spo2_from_ppg(
    ppg_signal,        # PPG signal array
    sampling_rate=50,  # Sample rate (Hz)
    time_interval=0.4  # Peak detection interval (seconds)
)
print(f"SpO2: {spo2:.2f}%, AC/DC Ratio: {ratio:.4f}")
```

**Empirical Formula**:
```python
spo2 = (-3.7465271198e+01 * ratio**3 +
         5.8403912586e+01 * ratio**2 +
        -3.7079378855e+01 * ratio +
         1.0016136403e+02)
spo2 = clip(spo2, 90, 100)  # Clip to 90-100% range
```

### Dataset Processing Modules

#### rw-ppg/rw_ppg.py - RW-PPG Dataset Processing

Processes the RW-PPG (Reflectance Wearable PPG) dataset, calculates and analyzes SpO2 values.

**Dataset Format**:
- Training set: 1374 signals, 300 samples, 50 Hz
- Test set: 700 signals, 300 samples, 50 Hz
- File format: Excel (.xlsx)

**Main Features**:
- Read training and test set Excel files
- Signal visualization (8-channel subplot display)
- Batch calculate SpO2 values for all signals
- Generate statistical analysis reports
- Output to Excel file (containing Training_Set, Test_Set, Statistics three sheets)
- Plot SpO2 distribution and box plots

**Output Files**:
- `rw-ppg/rw_ppg_signals.png` - Signal sample plot
- `rw-ppg/rw_ppg_spo2_data.xlsx` - SpO2 data
- `rw-ppg/rw_ppg_spo2_analysis.png` - SpO2 analysis plot

#### ppg-bp/ppg-bp.py - PPG-BP Dataset Processing

Batch processes all PPG signal files in the PPG-BP dataset.

**Dataset Format**:
- Sample rate: 1000 Hz
- File format: Plain text (.txt), one signal per line
- Automatically scans all txt files in the directory

**Main Features**:
- Automatically scan and sort all txt files in the directory
- Read each signal by line (tab-separated)
- Batch calculate SpO2 values and AC/DC ratios
- Generate complete statistical reports
- Plot signal samples and SpO2 analysis charts

**Output Files**:
- `ppg-bp/ppg_bp_spo2_data.xlsx` - SpO2 batch calculation results

#### but-ppg/but_ppg.py - BUT-PPG Dataset Reading

Reads WFDB format files from the BUT-PPG dataset.

**Dataset Format**:
- File format: WFDB format (.dat + .hea header file)
- Naming rule: `<record_name>_PPG.dat` / `<record_name>_PPG.hea`

**Main Features**:
- Automatically scan and extract all record names
- Use `wfdb.rdrecord()` to read signal data
- Use `wfdb.rdheader()` to read header information
- Display dataset structure information

### Utility Modules

#### concat_dataset.py - Dataset Concatenation Tool

Cyclically concatenates short PPG signals to generate longer test data.

**Main Features**:
- Read specified PPG file
- Cyclically concatenate raw data N times (default 8 times)
- Save concatenated data to new file
- Plot concatenated signal waveform

**Usage Scenarios**:
- Extend short-time signals to long-time test data
- Verify stability of real-time filters
- Test algorithm performance on long-duration data

**Output Files**:
- `concat_<record_name>.txt` - Concatenated data
- `concat_<record_name>.png` - Waveform plot

#### show.py - Signal Visualization Comparison

Compares differences between C++ and Python filtering results for algorithm validation.

**Main Features**:
- Read C++ filter output file
- Read Python filter output file
- Plot waveform comparison chart
- Calculate and display error statistics
- Verify correctness of C++ implementation

### Python Environment Dependencies

```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
pandas>=1.3.0
openpyxl>=3.0.0
wfdb>=3.4.0
```

Install dependencies:
```bash
pip install numpy scipy matplotlib pandas openpyxl wfdb
```

### Usage Examples

**Process RW-PPG dataset**:
```bash
cd aaaPyTest/rw-ppg
python rw_ppg.py
```

**Process PPG-BP dataset**:
```bash
cd aaaPyTest/ppg-bp
python ppg-bp.py
```

**Concatenate dataset**:
```bash
cd aaaPyTest
# Edit concat_dataset.py to modify record_name and concatenation multiplier
python concat_dataset.py
```
