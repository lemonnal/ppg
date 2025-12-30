import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 添加父目录到Python路径，以便导入Method模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from Method import calculate_spo2_from_ppg

# 设置字体（不需要中文字体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

folder_path = '/home/yogsothoth/桌面/workspace-ppg/DataSet/PPG-BP/'

# 自动扫描文件夹中的所有txt文件
text_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
text_files.sort()  # 按文件名排序

print(f"发现 {len(text_files)} 个txt文件:")
for f in text_files:
    print(f"  - {f}")

# 存储所有结果
all_signals = []
all_spo2_values = []
all_ratios = []
all_file_names = []

sampling_rate = 1000  # Hz
time_interval = 0.4  # 秒

print("="*60)
print("开始读取和处理PPG-BP数据...")
print("="*60)

# 读取数据并计算SpO2
for text_file in text_files:
    file_path = os.path.join(folder_path, text_file)
    print(f"\n正在处理文件: {text_file}")
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        print(f"  文件包含 {len(lines)} 条信号")
        
        for idx, line in enumerate(lines):
            signal = np.array([float(x.strip()) for x in line.split()])
            print(len(signal))
            spo2, ratio = calculate_spo2_from_ppg(signal, sampling_rate=sampling_rate, time_interval=time_interval)
            
            # 存储数据
            all_signals.append(signal)
            all_spo2_values.append(spo2)
            all_ratios.append(ratio)
            all_file_names.append(f"{text_file}_Signal{idx+1}")
            
            if idx < 10:  # 显示前10个
                print(f"    信号 {idx+1}: SpO2 = {spo2:.2f}%, AC/DC ratio = {ratio:.4f}")

# 转换为numpy数组
all_spo2_array = np.array(all_spo2_values)
all_ratios_array = np.array(all_ratios)

print(f"\n数据处理完成!")
print(f"  总信号数: {len(all_signals)}")
print(f"  每条信号采样点数: {len(all_signals[0]) if len(all_signals) > 0 else 0}")
print(f"  采样率: {sampling_rate} 样本/秒")
print(f"  信号时长: {len(all_signals[0])/sampling_rate:.2f} 秒" if len(all_signals) > 0 else "")

# ============= 统计信息 =============
print("\n" + "="*60)
print("SpO2统计信息:")
print("="*60)
print(f"  平均值: {all_spo2_array.mean():.2f}%")
print(f"  标准差: {all_spo2_array.std():.2f}%")
print(f"  最小值: {all_spo2_array.min():.2f}%")
print(f"  最大值: {all_spo2_array.max():.2f}%")
print(f"  中位数: {np.median(all_spo2_array):.2f}%")

print(f"\nAC/DC比率统计:")
print(f"  平均值: {all_ratios_array.mean():.4f}")
print(f"  标准差: {all_ratios_array.std():.4f}")
print(f"  最小值: {all_ratios_array.min():.4f}")
print(f"  最大值: {all_ratios_array.max():.4f}")

# ============= 保存数据到Excel =============
if len(all_signals) > 0:
    # 创建DataFrame
    results_df = pd.DataFrame({
        'Signal_ID': all_file_names,
        'SpO2(%)': all_spo2_values,
        'AC_DC_Ratio': all_ratios
    })
    
    # 统计信息DataFrame
    stats_data = {
        'Metric': ['Sample Count', 'Mean SpO2(%)', 'Std SpO2(%)', 'Min SpO2(%)', 
                   'Max SpO2(%)', 'Median SpO2(%)', 'Mean AC/DC Ratio'],
        'Value': [len(all_spo2_array), all_spo2_array.mean(), all_spo2_array.std(),
                  all_spo2_array.min(), all_spo2_array.max(), np.median(all_spo2_array),
                  all_ratios_array.mean()]
    }
    stats_df = pd.DataFrame(stats_data)
    
    # 保存到Excel
    excel_path = os.path.join(os.path.dirname(__file__), 'ppg_bp_spo2_data.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Results', index=False)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    print(f"\n数据已保存到: {excel_path}")
    print("  - Results sheet: 所有信号的SpO2值")
    print("  - Statistics sheet: 统计信息")

# ============= 绘制信号样本图 ============= 
if len(all_signals) > 0:
    # 选择要显示的信号数量（最多8条）
    num_display = min(8, len(all_signals))
    
    fig_signals, axes = plt.subplots(4, 2, figsize=(15, 12))
    fig_signals.suptitle('PPG-BP Signal Samples', fontsize=16, fontweight='bold')
    
    for i in range(num_display):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        signal = all_signals[i]
        duration = len(signal) / sampling_rate
        time_axis = np.linspace(0, duration, len(signal))
        
        ax.plot(time_axis, signal, linewidth=1.5, color='blue')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('PPG Amplitude', fontsize=10)
        ax.set_title(f'{all_file_names[i]}\nSpO2: {all_spo2_values[i]:.2f}%', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # 如果信号数少于8个，隐藏多余的子图
    for i in range(num_display, 8):
        row = i // 2
        col = i % 2
        axes[row, col].axis('off')
    
    plt.tight_layout()
    print("\n显示信号样本图...")
    plt.show()

# ============= 绘制SpO2分析图 ============= 
if len(all_signals) > 0:
    fig_spo2, axes_spo2 = plt.subplots(2, 2, figsize=(14, 10))
    fig_spo2.suptitle('SpO2 Analysis (PPG-BP Dataset)', fontsize=16, fontweight='bold')
    
    # 1. SpO2分布直方图
    axes_spo2[0, 0].hist(all_spo2_array, bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes_spo2[0, 0].axvline(all_spo2_array.mean(), color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {all_spo2_array.mean():.2f}%')
    axes_spo2[0, 0].set_xlabel('SpO2 (%)')
    axes_spo2[0, 0].set_ylabel('Frequency')
    axes_spo2[0, 0].set_title(f'SpO2 Distribution (n={len(all_spo2_array)})')
    axes_spo2[0, 0].legend()
    axes_spo2[0, 0].grid(True, alpha=0.3)
    
    # 2. AC/DC比率分布直方图
    axes_spo2[0, 1].hist(all_ratios_array, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes_spo2[0, 1].axvline(all_ratios_array.mean(), color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {all_ratios_array.mean():.4f}')
    axes_spo2[0, 1].set_xlabel('AC/DC Ratio')
    axes_spo2[0, 1].set_ylabel('Frequency')
    axes_spo2[0, 1].set_title(f'AC/DC Ratio Distribution (n={len(all_ratios_array)})')
    axes_spo2[0, 1].legend()
    axes_spo2[0, 1].grid(True, alpha=0.3)
    
    # 3. SpO2箱线图
    box = axes_spo2[1, 0].boxplot([all_spo2_array], labels=['PPG-BP'], patch_artist=True)
    box['boxes'][0].set_facecolor('blue')
    axes_spo2[1, 0].set_ylabel('SpO2 (%)')
    axes_spo2[1, 0].set_title('SpO2 Box Plot')
    axes_spo2[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息文本
    stats_text = f'Mean: {all_spo2_array.mean():.2f}%\n'
    stats_text += f'Std: {all_spo2_array.std():.2f}%\n'
    stats_text += f'Min: {all_spo2_array.min():.2f}%\n'
    stats_text += f'Max: {all_spo2_array.max():.2f}%\n'
    stats_text += f'Median: {np.median(all_spo2_array):.2f}%'
    axes_spo2[1, 0].text(1.15, all_spo2_array.mean(), stats_text,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                         fontsize=9)
    
    # 4. SpO2随信号序号变化
    axes_spo2[1, 1].plot(range(1, len(all_spo2_array)+1), all_spo2_array, 
                         marker='o', linestyle='-', markersize=4, linewidth=1, color='purple')
    axes_spo2[1, 1].axhline(all_spo2_array.mean(), color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {all_spo2_array.mean():.2f}%')
    axes_spo2[1, 1].set_xlabel('Signal Index')
    axes_spo2[1, 1].set_ylabel('SpO2 (%)')
    axes_spo2[1, 1].set_title('SpO2 Values by Signal')
    axes_spo2[1, 1].legend()
    axes_spo2[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("显示SpO2分析图...")
    plt.show()

print("\n" + "="*60)
print("处理完成！")
print("="*60)
