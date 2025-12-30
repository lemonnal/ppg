import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import wfdb

# 添加父目录到Python路径，以便导入Method模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from Method import calculate_spo2_from_ppg

# 设置字体（不需要中文字体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

folder_path = '/home/yogsothoth/桌面/workspace-ppg/DataSet/PPG-BP/'
text_files = ['2_1.txt']

# 存储所有结果用于画图
all_signals = []
all_spo2_values = []
all_ratios = []
all_file_names = []

for text_file in text_files:
    file_path = os.path.join(folder_path, text_file)
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            signal = np.array([float(x.strip()) for x in line.split()])
            spo2, ratio = calculate_spo2_from_ppg(signal, 50)
            print(f"文件: {text_file}, 行{idx+1}: SpO2={spo2:.2f}%, Ratio={ratio:.4f}")
            
            # 存储数据用于画图
            all_signals.append(signal)
            all_spo2_values.append(spo2)
            all_ratios.append(ratio)
            all_file_names.append(f"{text_file}_行{idx+1}")

# 绘制结果
if len(all_signals) > 0:
    # 创建图形
    fig = plt.figure(figsize=(15, 10))
    
    # 计算需要多少个子图
    n_signals = len(all_signals)
    
    # 绘制每个信号
    for i in range(n_signals):
        # 信号图
        ax1 = plt.subplot(n_signals, 2, 2*i+1)
        time = np.arange(len(all_signals[i])) / 50  # 采样率50Hz
        ax1.plot(time, all_signals[i], 'b-', linewidth=1)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('PPG Signal')
        ax1.set_title(f'{all_file_names[i]} - PPG Signal')
        ax1.grid(True, alpha=0.3)
        
        # SpO2结果图
        ax2 = plt.subplot(n_signals, 2, 2*i+2)
        ax2.barh(['SpO2'], [all_spo2_values[i]], color='green' if all_spo2_values[i] >= 95 else 'orange')
        ax2.set_xlim([90, 100])
        ax2.set_xlabel('SpO2 (%)')
        ax2.set_title(f'{all_file_names[i]} - SpO2: {all_spo2_values[i]:.2f}%, Ratio: {all_ratios[i]:.4f}')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 在柱状图上添加数值标签
        ax2.text(all_spo2_values[i], 0, f'{all_spo2_values[i]:.2f}%', 
                ha='center', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    # # 保存图片
    # output_path = os.path.join(os.path.dirname(__file__), 'ppg_bp_analysis.png')
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # print(f"\n图片已保存到: {output_path}")
    
    plt.show()
else:
    print("没有数据可以绘制！")




