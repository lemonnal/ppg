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

# 读取训练数据
# 数据结构说明：
# 第1-300行：PPG信号数据（每列是一条信号）
# 第302行：标签"Serial"
# 第303行：序列号
# 第304行：标签"ID"
# 第305行：ID号
# 所有列都是信号数据

file_path1 = '/home/yogsothoth/桌面/workspace-ppg/DataSet/RW-PPG/train8.xlsx'
file_path2 = '/home/yogsothoth/桌面/workspace-ppg/DataSet/RW-PPG/test8.xlsx'

# 只读取前300行的信号数据
data1 = pd.read_excel(file_path1, header=None, nrows=300)
data2 = pd.read_excel(file_path2, header=None, nrows=300)

print("数据集形状:", data1.shape, data2.shape)
print("\n前5行、前5列数据预览:")
print(data1.iloc[:5, :5])

# 数据结构：每列是一条信号，每行是一个采样点
# 训练集: (300, 1374) - 300个采样点，1374条信号
# 测试集: (300, 700) - 300个采样点，700条信号

# 创建时间轴
sampling_rate = 50  # 样本/秒
num_samples = data1.shape[0]  # 300个采样点
duration = num_samples / sampling_rate  # 6.0秒
time_axis = np.linspace(0, duration, num_samples)

print(f"\n数据信息:")
print(f"- 训练集: {data1.shape[1]} 条信号")
print(f"- 测试集: {data2.shape[1]} 条信号")
print(f"- 每条信号采样点数: {num_samples}")
print(f"- 采样率: {sampling_rate} 样本/秒")
print(f"- 信号时长: {duration:.2f} 秒")
print(f"- 300个采样点, 6秒记录时长, 50样本/秒")

# 绘制多个信号样本（从训练集中选择）
fig, axes = plt.subplots(4, 2, figsize=(15, 12))
fig.suptitle('PPG Signal Samples (Training Set)', fontsize=16, fontweight='bold')

# 选择前8条信号进行可视化
for i in range(8):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    # 获取第i列的信号数据（每列是一条完整信号）
    signal = data1.iloc[:, i].values
    
    ax.plot(time_axis, signal, linewidth=1.5, color='blue')
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('PPG Amplitude', fontsize=10)
    ax.set_title(f'Signal #{i+1}', fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rw-ppg/rw_ppg_signals.png', dpi=300, bbox_inches='tight')
print("\n图表已保存为 'rw-ppg/rw_ppg_signals.png'")
plt.show()


# ============= 计算每条信号的SpO2值 =============
print("\n" + "="*60)
print("开始计算SpO2值...")
print("="*60)

# 计算所有信号的SpO2
print("\n计算训练集SpO2...")
train_spo2_list = []
train_ratio_list = []

for i in range(data1.shape[1]):
    signal = data1.iloc[:, i].values
    spo2, ratio = calculate_spo2_from_ppg(signal, sampling_rate)
    train_spo2_list.append(spo2)
    train_ratio_list.append(ratio)
    
    if i < 10:  # 显示前10个
        print(f"  信号 {i+1}: SpO2 = {spo2:.2f}%, AC/DC ratio = {ratio:.4f}")

print("\n计算测试集SpO2...")
test_spo2_list = []
test_ratio_list = []

for i in range(data2.shape[1]):
    signal = data2.iloc[:, i].values
    spo2, ratio = calculate_spo2_from_ppg(signal)
    test_spo2_list.append(spo2)
    test_ratio_list.append(ratio)
    
    if i < 10:  # 显示前10个
        print(f"  信号 {i+1}: SpO2 = {spo2:.2f}%, AC/DC ratio = {ratio:.4f}")

# 转换为numpy数组
train_spo2 = np.array(train_spo2_list)
test_spo2 = np.array(test_spo2_list)

print(f"\n训练集SpO2统计:")
print(f"  平均值: {train_spo2.mean():.2f}%")
print(f"  标准差: {train_spo2.std():.2f}%")
print(f"  最小值: {train_spo2.min():.2f}%")
print(f"  最大值: {train_spo2.max():.2f}%")

print(f"\n测试集SpO2统计:")
print(f"  平均值: {test_spo2.mean():.2f}%")
print(f"  标准差: {test_spo2.std():.2f}%")
print(f"  最小值: {test_spo2.min():.2f}%")
print(f"  最大值: {test_spo2.max():.2f}%")

# 保存SpO2数据到Excel文件
# 创建训练集DataFrame
train_df = pd.DataFrame({
    'Signal_ID': range(1, len(train_spo2) + 1),
    'SpO2(%)': train_spo2,
    'AC_DC_Ratio': train_ratio_list
})

# 创建测试集DataFrame
test_df = pd.DataFrame({
    'Signal_ID': range(1, len(test_spo2) + 1),
    'SpO2(%)': test_spo2,
    'AC_DC_Ratio': test_ratio_list
})

# 保存到Excel文件（多个sheet）
with pd.ExcelWriter('rw-ppg/rw_ppg_spo2_data.xlsx', engine='openpyxl') as writer:
    train_df.to_excel(writer, sheet_name='Training_Set', index=False)
    test_df.to_excel(writer, sheet_name='Test_Set', index=False)
    
    # 添加统计信息sheet
    stats_data = {
        'Dataset': ['Training Set', 'Test Set'],
        'Sample_Count': [len(train_spo2), len(test_spo2)],
        'Mean_SpO2(%)': [train_spo2.mean(), test_spo2.mean()],
        'Std_SpO2(%)': [train_spo2.std(), test_spo2.std()],
        'Min_SpO2(%)': [train_spo2.min(), test_spo2.min()],
        'Max_SpO2(%)': [train_spo2.max(), test_spo2.max()],
        'Mean_AC_DC_Ratio': [np.mean(train_ratio_list), np.mean(test_ratio_list)]
    }
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_excel(writer, sheet_name='Statistics', index=False)

print("\nSpO2数据已保存到 'rw-ppg/rw_ppg_spo2_data.xlsx'")
print("  - Training_Set sheet: 训练集数据 (1374条)")
print("  - Test_Set sheet: 测试集数据 (700条)")
print("  - Statistics sheet: 统计信息")

# 绘制SpO2分布图
fig_spo2, axes_spo2 = plt.subplots(2, 2, figsize=(14, 10))
fig_spo2.suptitle('SpO2 Analysis', fontsize=16, fontweight='bold')

# 1. 训练集SpO2分布直方图
axes_spo2[0, 0].hist(train_spo2, bins=30, color='blue', alpha=0.7, edgecolor='black')
axes_spo2[0, 0].axvline(train_spo2.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {train_spo2.mean():.2f}%')
axes_spo2[0, 0].set_xlabel('SpO2 (%)')
axes_spo2[0, 0].set_ylabel('Frequency')
axes_spo2[0, 0].set_title(f'Training Set SpO2 Distribution (n={len(train_spo2)})')
axes_spo2[0, 0].legend()
axes_spo2[0, 0].grid(True, alpha=0.3)

# 2. 测试集SpO2分布直方图
axes_spo2[0, 1].hist(test_spo2, bins=30, color='green', alpha=0.7, edgecolor='black')
axes_spo2[0, 1].axvline(test_spo2.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {test_spo2.mean():.2f}%')
axes_spo2[0, 1].set_xlabel('SpO2 (%)')
axes_spo2[0, 1].set_ylabel('Frequency')
axes_spo2[0, 1].set_title(f'Test Set SpO2 Distribution (n={len(test_spo2)})')
axes_spo2[0, 1].legend()
axes_spo2[0, 1].grid(True, alpha=0.3)

# 3. 训练集和测试集SpO2对比
axes_spo2[1, 0].hist(train_spo2, bins=30, color='blue', alpha=0.5, label='Training', density=True)
axes_spo2[1, 0].hist(test_spo2, bins=30, color='green', alpha=0.5, label='Test', density=True)
axes_spo2[1, 0].set_xlabel('SpO2 (%)')
axes_spo2[1, 0].set_ylabel('Probability Density')
axes_spo2[1, 0].set_title('SpO2 Distribution Comparison')
axes_spo2[1, 0].legend()
axes_spo2[1, 0].grid(True, alpha=0.3)

# 4. SpO2箱线图对比
box_data = [train_spo2, test_spo2]
box = axes_spo2[1, 1].boxplot(box_data, labels=['Training', 'Test'], patch_artist=True)
box['boxes'][0].set_facecolor('blue')
box['boxes'][1].set_facecolor('green')
axes_spo2[1, 1].set_ylabel('SpO2 (%)')
axes_spo2[1, 1].set_title('SpO2 Box Plot Comparison')
axes_spo2[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('rw-ppg/rw_ppg_spo2_analysis.png', dpi=300, bbox_inches='tight')
print("SpO2分析图表已保存为 'rw-ppg/rw_ppg_spo2_analysis.png'")
plt.show()

print("\n" + "="*60)
print("SpO2计算完成！")
print("="*60)