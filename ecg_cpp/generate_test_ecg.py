#!/usr/bin/env python3
"""
生成测试ECG信号

生成一个简单的合成ECG信号用于测试QRS检测器
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_ecg(duration=10, fs=360, hr=72):
    """
    生成合成ECG信号
    
    参数:
        duration: 信号时长（秒）
        fs: 采样率 (Hz)
        hr: 心率 (bpm)
    
    返回:
        signal: 合成的ECG信号
        true_peaks: 真实的R峰位置
    """
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros(len(t))
    
    # 计算RR间期（秒）
    rr_interval = 60.0 / hr
    
    # 生成R峰位置
    true_peaks = []
    current_time = 0.5  # 从0.5秒开始
    
    while current_time < duration - 0.5:
        peak_idx = int(current_time * fs)
        true_peaks.append(peak_idx)
        
        # 添加一些心率变异性（±5%）
        variation = np.random.uniform(0.95, 1.05)
        current_time += rr_interval * variation
    
    # 为每个R峰生成QRS波形
    for peak_idx in true_peaks:
        # QRS波复合体
        qrs_duration = int(0.08 * fs)  # QRS持续时间约80ms
        
        # Q波（小的负向波）
        q_start = peak_idx - int(0.02 * fs)
        q_end = peak_idx - int(0.01 * fs)
        if q_start >= 0 and q_end < len(signal):
            q_wave = -0.1 * np.sin(np.linspace(0, np.pi, q_end - q_start))
            signal[q_start:q_end] += q_wave
        
        # R波（主要的正向波）
        r_start = peak_idx - int(0.01 * fs)
        r_end = peak_idx + int(0.01 * fs)
        if r_start >= 0 and r_end < len(signal):
            r_wave = 1.0 * np.sin(np.linspace(0, np.pi, r_end - r_start))
            signal[r_start:r_end] += r_wave
        
        # S波（负向波）
        s_start = peak_idx + int(0.01 * fs)
        s_end = peak_idx + int(0.04 * fs)
        if s_start >= 0 and s_end < len(signal):
            s_wave = -0.3 * np.sin(np.linspace(0, np.pi, s_end - s_start))
            signal[s_start:s_end] += s_wave
        
        # P波（R峰前的正向波）
        p_center = peak_idx - int(0.18 * fs)
        p_width = int(0.08 * fs)
        p_start = max(0, p_center - p_width // 2)
        p_end = min(len(signal), p_center + p_width // 2)
        if p_start >= 0 and p_end < len(signal):
            p_wave = 0.15 * np.sin(np.linspace(0, np.pi, p_end - p_start))
            signal[p_start:p_end] += p_wave
        
        # T波（R峰后的正向波）
        t_center = peak_idx + int(0.20 * fs)
        t_width = int(0.12 * fs)
        t_start = max(0, t_center - t_width // 2)
        t_end = min(len(signal), t_center + t_width // 2)
        if t_start >= 0 and t_end < len(signal):
            t_wave = 0.25 * np.sin(np.linspace(0, np.pi, t_end - t_start))
            signal[t_start:t_end] += t_wave
    
    # 添加基线漂移（低频成分）
    baseline_freq = 0.3  # Hz
    baseline = 0.1 * np.sin(2 * np.pi * baseline_freq * t)
    signal += baseline
    
    # 添加噪声
    noise_level = 0.03
    noise = np.random.normal(0, noise_level, len(signal))
    signal += noise
    
    return signal, true_peaks


def save_signal(signal, filename):
    """保存信号到文件"""
    with open(filename, 'w') as f:
        for value in signal:
            f.write(f"{value}\n")
    print(f"信号已保存: {filename}")


def save_peaks(peaks, filename):
    """保存峰值位置到文件"""
    with open(filename, 'w') as f:
        for peak in peaks:
            f.write(f"{peak}\n")
    print(f"峰值位置已保存: {filename}")


def visualize_signal(signal, true_peaks, fs=360, duration=10):
    """可视化信号"""
    samples = int(duration * fs)
    signal_segment = signal[:samples]
    time = np.arange(len(signal_segment)) / fs
    
    # 筛选在显示范围内的峰值
    peaks_segment = [p for p in true_peaks if p < samples]
    
    plt.figure(figsize=(15, 6))
    plt.plot(time, signal_segment, 'b-', linewidth=1, label='合成ECG信号')
    
    # 标记R峰
    peak_times = [p / fs for p in peaks_segment]
    peak_values = [signal[p] for p in peaks_segment]
    plt.plot(peak_times, peak_values, 'ro', markersize=8, label=f'R峰 ({len(peaks_segment)}个)')
    
    plt.xlabel('时间 (秒)', fontsize=12)
    plt.ylabel('幅值', fontsize=12)
    plt.title('合成ECG信号', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图形
    import os
    os.makedirs('output_signal_figure', exist_ok=True)
    plt.savefig('output_signal_figure/synthetic_ecg.png', dpi=150)
    print("可视化图已保存: output_signal_figure/synthetic_ecg.png")
    
    plt.show()


def main():
    print("="*70)
    print("生成合成ECG测试信号")
    print("="*70)
    
    # 参数设置
    duration = 30  # 30秒
    fs = 360       # 360 Hz采样率（与MIT-BIH相同）
    hr = 72        # 72 bpm心率
    
    print(f"\n参数配置:")
    print(f"  时长: {duration} 秒")
    print(f"  采样率: {fs} Hz")
    print(f"  心率: {hr} bpm")
    
    # 生成信号
    print("\n正在生成信号...")
    signal, true_peaks = generate_synthetic_ecg(duration, fs, hr)
    
    print(f"\n生成完成:")
    print(f"  信号长度: {len(signal)} 样本")
    print(f"  R峰数量: {len(true_peaks)}")
    print(f"  平均心率: {len(true_peaks) / duration * 60:.1f} bpm")
    
    # 保存信号
    print("\n保存文件...")
    import os
    os.makedirs('output_data', exist_ok=True)
    
    signal_file = 'output_data/test_ecg_signal.txt'
    peaks_file = 'output_data/test_ecg_peaks_true.txt'
    
    save_signal(signal, signal_file)
    save_peaks(true_peaks, peaks_file)
    
    # 可视化
    print("\n生成可视化...")
    visualize_signal(signal, true_peaks, fs, duration=10)
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"\n可以使用以下命令测试:")
    print(f"  Python版本:")
    print(f"    cd ecg_cpp && python offline.py")
    print(f"  C++版本:")
    print(f"    ./build/ecg_offline_main {signal_file} MLII 360")
    print(f"  对比测试:")
    print(f"    python compare_python_cpp.py {signal_file} MLII 360")


if __name__ == "__main__":
    main()

