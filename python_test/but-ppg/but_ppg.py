import os
import numpy as np
import wfdb

# BUT-PPG数据集路径
folder_path = '/home/yogsothoth/桌面/workspace-ppg/DataSet/BUT-PPG/'

# 获取所有文件
all_files = os.listdir(folder_path)

# 提取记录名（去除_PPG.dat和_PPG.hea后缀）
record_names = set()
for file in all_files:
    if file.endswith('_PPG.dat'):
        record_name = file.replace('_PPG.dat', '')
        record_names.add(record_name)
    elif file.endswith('_PPG.hea'):
        record_name = file.replace('_PPG.hea', '')
        record_names.add(record_name)

# 排序
record_names = sorted(list(record_names))

data = wfdb.rdrecord(folder_path + record_names[0] + '_PPG')
head = wfdb.rdheader(folder_path + record_names[0] + '_PPG')
for key, value in data.__dict__.items():
    print(key, value)