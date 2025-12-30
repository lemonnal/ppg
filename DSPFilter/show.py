#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPG信号对比图 - 原始信号 vs 滤波后信号
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置开关 ====================
USE_PHASE_COMPENSATION = False  # 是否使用相位延迟补偿（True=对齐波形, False=不对齐）
# ================================================

# 文件路径
original_file = '/home/yogsothoth/桌面/workspace-ppg/DataSet/PPG-BP/2_1.txt'
# filtered_file = '/home/yogsothoth/桌面/workspace-ppg/DSPFilter/PPG-BP/2_1_filtered_zerophase.txt'
filtered_file = '/home/yogsothoth/桌面/workspace-ppg/DSPFilter/PPG-BP/2_1_filtered_oneway.txt'

print("=" * 60)
print("读取原始PPG数据...")
print("=" * 60)

# 读取原始数据（每行包含多个值，用制表符或空格分隔）
with open(original_file, 'r') as f:
    lines = f.readlines()

# 解析第一行数据（假设我们处理第一条信号）
original_data = []
for line in lines:
    # 分割数据并转换为浮点数
    values = [float(x.strip()) for x in line.split() if x.strip()]
    original_data.extend(values)
    break  # 只读取第一行

original_data = np.array(original_data)
print(f"原始数据长度: {len(original_data)} 个采样点")
print(f"原始数据范围: [{original_data.min():.2f}, {original_data.max():.2f}]")

print("\n" + "=" * 60)
print("读取滤波后数据...")
print("=" * 60)

# 读取滤波后的数据（每行一个值）
filtered_data = []
with open(filtered_file, 'r') as f:
    for line in f:
        if line.strip():
            filtered_data.append(float(line.strip()))

filtered_data = np.array(filtered_data)
print(f"滤波后数据长度: {len(filtered_data)} 个采样点")
print(f"滤波后数据范围: [{filtered_data.min():.2f}, {filtered_data.max():.2f}]")

# 采样率 (Hz)
sampling_rate = 1000  # PPG-BP数据集采样率为1000Hz

# ==================== 相位延迟补偿 ====================
if USE_PHASE_COMPENSATION:
    print("\n" + "=" * 60)
    print("应用相位延迟补偿...")
    print("=" * 60)
    
    # 对于3阶Butterworth带通（0.5-20Hz），估算群延迟
    filter_order = 3
    low_freq = 0.5
    delay_samples = int(filter_order / (2 * np.pi * low_freq) * sampling_rate * 0.025)  # 约23ms的延迟
    
    print(f"  延迟样本数: {delay_samples} 个样本 ({delay_samples/sampling_rate*1000:.1f} ms)")
    print(f"  操作: 滤波信号左移 {delay_samples} 个样本以对齐原始信号")
    
    # 左移滤波信号
    filtered_data_shifted = filtered_data[delay_samples:]
    original_data_shifted = original_data[:len(filtered_data_shifted)]
    
    # 更新时间轴
    time = np.arange(len(filtered_data_shifted)) / sampling_rate
    
    compensation_status = f"Phase Aligned (shifted {delay_samples} samples)"
else:
    print("\n" + "=" * 60)
    print("跳过相位延迟补偿 (显示原始延迟)")
    print("=" * 60)
    
    # 不进行相位补偿，统一长度
    min_length = min(len(original_data), len(filtered_data))
    filtered_data_shifted = filtered_data[:min_length]
    original_data_shifted = original_data[:min_length]
    
    # 更新时间轴
    time = np.arange(min_length) / sampling_rate
    
    compensation_status = "Original (with phase delay)"

print("\n" + "=" * 60)
print("生成对比图形...")
print("=" * 60)

# 创建图形
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 子图1: 原始信号
axes[0].plot(time, original_data_shifted, 'b-', linewidth=0.8, alpha=0.7)
axes[0].set_title('Original PPG Signal (2_1.txt)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Time (s)', fontsize=11)
axes[0].set_ylabel('Amplitude', fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, time[-1]])

# 子图2: 滤波后信号
axes[1].plot(time, filtered_data_shifted, 'r-', linewidth=0.8, alpha=0.7)
axes[1].set_title(f'Filtered PPG Signal ({compensation_status})', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Time (s)', fontsize=11)
axes[1].set_ylabel('Amplitude', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, time[-1]])

# 子图3: 叠加对比
axes[2].plot(time, original_data_shifted, 'b-', linewidth=0.8, alpha=0.5, label='Original')
filtered_label = 'Filtered (aligned)' if USE_PHASE_COMPENSATION else 'Filtered (with delay)'
axes[2].plot(time, filtered_data_shifted, 'r-', linewidth=1.2, alpha=0.8, label=filtered_label)
title_suffix = 'Phase Aligned' if USE_PHASE_COMPENSATION else 'With Phase Delay'
axes[2].set_title(f'Comparison: Original vs Filtered ({title_suffix})', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Amplitude', fontsize=11)
axes[2].legend(loc='upper right', fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([0, time[-1]])

# 调整布局
plt.tight_layout()

# # 保存图形
# output_path = '/home/yogsothoth/桌面/workspace-ppg/DSPFilter/PPG-BP/comparison.png'
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"\n图形已保存到: {output_path}")

# 显示图形
plt.show()

print("\n" + "=" * 60)
print("完成!")
print("=" * 60)

# 打印统计信息
print("\n统计信息:")
print(f"  原始信号 - 均值: {original_data_shifted.mean():.2f}, 标准差: {original_data_shifted.std():.2f}")
print(f"  滤波信号 - 均值: {filtered_data_shifted.mean():.2f}, 标准差: {filtered_data_shifted.std():.2f}")

