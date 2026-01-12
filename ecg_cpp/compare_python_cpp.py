#!/usr/bin/env python3
"""
对比Python和C++版本的QRS检测结果

使用方法:
    python compare_python_cpp.py <信号文件> [导联名称] [采样率]

功能:
    1. 使用Python版本检测QRS波
    2. 调用C++版本检测QRS波
    3. 对比两个版本的结果
    4. 生成可视化对比图
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess

# 添加ecg_cpp目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ecg_cpp'))

from ecg_cpp.offline import PanTomkinsQRSDetectorOffline


def read_signal_file(filename):
    """读取信号文件"""
    with open(filename, 'r') as f:
        signal = [float(line.strip()) for line in f if line.strip()]
    return np.array(signal)


def read_peaks_file(filename):
    """读取峰值文件"""
    with open(filename, 'r') as f:
        peaks = [int(line.strip()) for line in f if line.strip()]
    return peaks


def run_cpp_detector(signal_file, signal_name="MLII", fs=360):
    """运行C++版本的检测器"""
    cpp_executable = "./build/ecg_offline_main"
    
    if not os.path.exists(cpp_executable):
        print("错误: C++可执行文件不存在，请先编译")
        print("运行: ./build_and_run_ecg_offline.sh")
        return None
    
    # 运行C++程序
    cmd = [cpp_executable, signal_file, signal_name, str(fs), "0"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("C++检测器运行成功")
        
        # 读取输出的峰值文件
        base_name = os.path.basename(signal_file).rsplit('.', 1)[0]
        peaks_file = f"output_data/{base_name}_peaks.txt"
        
        if os.path.exists(peaks_file):
            return read_peaks_file(peaks_file)
        else:
            print(f"警告: 找不到峰值文件 {peaks_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"C++检测器运行失败: {e}")
        print(f"错误输出: {e.stderr}")
        return None


def compare_peaks(py_peaks, cpp_peaks, tolerance=5):
    """
    对比两个峰值列表
    
    参数:
        py_peaks: Python版本检测的峰值
        cpp_peaks: C++版本检测的峰值
        tolerance: 允许的误差范围（样本点）
    """
    if py_peaks is None or cpp_peaks is None:
        print("错误: 无法对比，其中一个检测结果为空")
        return
    
    print("\n" + "="*70)
    print("检测结果对比")
    print("="*70)
    
    print(f"\nPython版本检测到: {len(py_peaks)} 个R峰")
    print(f"C++版本检测到: {len(cpp_peaks)} 个R峰")
    print(f"差异: {abs(len(py_peaks) - len(cpp_peaks))} 个峰")
    
    # 计算匹配度
    matched = 0
    unmatched_py = []
    unmatched_cpp = list(cpp_peaks)
    
    for py_peak in py_peaks:
        found = False
        for cpp_peak in cpp_peaks:
            if abs(py_peak - cpp_peak) <= tolerance:
                matched += 1
                if cpp_peak in unmatched_cpp:
                    unmatched_cpp.remove(cpp_peak)
                found = True
                break
        if not found:
            unmatched_py.append(py_peak)
    
    match_rate = (matched / max(len(py_peaks), len(cpp_peaks))) * 100 if max(len(py_peaks), len(cpp_peaks)) > 0 else 0
    
    print(f"\n匹配的峰值数量: {matched}")
    print(f"匹配率: {match_rate:.2f}%")
    print(f"Python独有峰值: {len(unmatched_py)}")
    print(f"C++独有峰值: {len(unmatched_cpp)}")
    
    # 显示前5个峰值的对比
    print("\n前5个峰值位置对比:")
    print(f"{'索引':<8} {'Python':<12} {'C++':<12} {'差异':<12}")
    print("-" * 50)
    for i in range(min(5, len(py_peaks), len(cpp_peaks))):
        diff = py_peaks[i] - cpp_peaks[i]
        print(f"{i+1:<8} {py_peaks[i]:<12} {cpp_peaks[i]:<12} {diff:<12}")
    
    return {
        'matched': matched,
        'match_rate': match_rate,
        'unmatched_py': unmatched_py,
        'unmatched_cpp': unmatched_cpp
    }


def visualize_comparison(signal, py_peaks, cpp_peaks, fs=360, duration=10):
    """
    可视化对比结果
    
    参数:
        signal: ECG信号
        py_peaks: Python版本检测的峰值
        cpp_peaks: C++版本检测的峰值
        fs: 采样率
        duration: 显示时长（秒）
    """
    # 只显示前duration秒的数据
    samples = int(duration * fs)
    signal_segment = signal[:samples]
    time = np.arange(len(signal_segment)) / fs
    
    # 筛选在显示范围内的峰值
    py_peaks_segment = [p for p in py_peaks if p < samples]
    cpp_peaks_segment = [p for p in cpp_peaks if p < samples]
    
    # 创建图形
    plt.figure(figsize=(15, 8))
    
    # 绘制信号
    plt.plot(time, signal_segment, 'b-', linewidth=0.5, alpha=0.7, label='ECG信号')
    
    # 绘制Python检测的峰值
    py_peak_times = [p / fs for p in py_peaks_segment]
    py_peak_values = [signal[p] for p in py_peaks_segment]
    plt.plot(py_peak_times, py_peak_values, 'ro', markersize=8, 
             label=f'Python检测 ({len(py_peaks_segment)}个)', alpha=0.7)
    
    # 绘制C++检测的峰值
    cpp_peak_times = [p / fs for p in cpp_peaks_segment]
    cpp_peak_values = [signal[p] for p in cpp_peaks_segment]
    plt.plot(cpp_peak_times, cpp_peak_values, 'g^', markersize=8, 
             label=f'C++检测 ({len(cpp_peaks_segment)}个)', alpha=0.7)
    
    plt.xlabel('时间 (秒)', fontsize=12)
    plt.ylabel('幅值', fontsize=12)
    plt.title('Python vs C++ QRS检测结果对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图形
    output_file = 'output_signal_figure/python_cpp_comparison.png'
    os.makedirs('output_signal_figure', exist_ok=True)
    plt.savefig(output_file, dpi=150)
    print(f"\n对比图已保存: {output_file}")
    
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("用法: python compare_python_cpp.py <信号文件> [导联名称] [采样率]")
        print("示例: python compare_python_cpp.py test_ecg.txt MLII 360")
        sys.exit(1)
    
    signal_file = sys.argv[1]
    signal_name = sys.argv[2] if len(sys.argv) > 2 else "MLII"
    fs = float(sys.argv[3]) if len(sys.argv) > 3 else 360.0
    
    if not os.path.exists(signal_file):
        print(f"错误: 文件不存在 {signal_file}")
        sys.exit(1)
    
    print("="*70)
    print("Python vs C++ QRS检测器对比")
    print("="*70)
    print(f"\n信号文件: {signal_file}")
    print(f"导联名称: {signal_name}")
    print(f"采样率: {fs} Hz")
    
    # 读取信号
    print("\n正在读取信号...")
    signal = read_signal_file(signal_file)
    print(f"信号长度: {len(signal)} 样本 ({len(signal)/fs:.2f} 秒)")
    
    # Python版本检测
    print("\n" + "-"*70)
    print("运行Python版本检测器...")
    print("-"*70)
    py_detector = PanTomkinsQRSDetectorOffline(signal_name=signal_name)
    py_detector.fs = fs
    py_peaks = py_detector.detect_qrs_peaks(signal)
    
    # 计算Python版本的心率统计
    py_stats = py_detector.get_heart_rate_statistics()
    print(f"\nPython检测结果:")
    print(f"  检测到 {len(py_peaks)} 个R峰")
    print(f"  平均心率: {py_stats['mean_hr']:.1f} bpm")
    print(f"  心率标准差: {py_stats['std_hr']:.1f} bpm")
    
    # C++版本检测
    print("\n" + "-"*70)
    print("运行C++版本检测器...")
    print("-"*70)
    cpp_peaks = run_cpp_detector(signal_file, signal_name, int(fs))
    
    if cpp_peaks is None:
        print("\nC++版本运行失败，无法完成对比")
        sys.exit(1)
    
    # 对比结果
    comparison = compare_peaks(py_peaks, cpp_peaks, tolerance=5)
    
    # 可视化
    print("\n正在生成可视化对比图...")
    visualize_comparison(signal, py_peaks, cpp_peaks, fs, duration=10)
    
    print("\n" + "="*70)
    print("对比完成！")
    print("="*70)


if __name__ == "__main__":
    main()

