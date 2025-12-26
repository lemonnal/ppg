import wfdb
import numpy as np
import os
from offline import PanTomkinsQRSDetectorOffline
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mode = 2
    if  mode == 1:
        # MIT-BIH - 单通道检测
        # 路径
        root = "/home/yogsothoth/DataSet/mit-bih-arrhythmia-database-1.0.0/"
        # numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
        #              '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
        #              '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
        #              '231', '232', '233', '234']
        numberSet = ['100']

        # 定义要检测的目标通道
        target_lead = "MLII"  # 可以修改为其他导联如 "V1", "V5" 等

        total_annotation_num = 0
        total_detection_num = 0

        # 详细统计数据结构
        detailed_stats = {
            "tp": 0, "fp": 0, "fn": 0, "time_errors": []
        }

        print(f"开始检测 {target_lead} 导联...")
        print("=" * 60)

        for num in numberSet:
            try:
                # 加载数据文件
                input_data = wfdb.rdrecord(root + num)
                sig_name = input_data.sig_name

                # 检查是否包含目标导联
                if target_lead not in sig_name:
                    continue

                # 获取目标导联的信号
                target_idx = sig_name.index(target_lead)
                signal = wfdb.rdrecord(root + num, channel_names=[target_lead]).p_signal.flatten()
                qrs_detector = PanTomkinsQRSDetectorOffline(signal_name=target_lead)
                filtered_signal = qrs_detector.bandpass_filter(signal)

                print(f"\n处理记录 {num} - {target_lead} 导联")
                print(f"信号长度: {len(signal)} 样本")
                
                # 将signal输出到 /home/yogsothoth/桌面/workspace-ppg/python_test/dataset/目录
                dataset_dir = "/home/yogsothoth/桌面/workspace-ppg/dataset"
                os.makedirs(dataset_dir, exist_ok=True)  # 创建目录（如果不存在）
                output_path_original = os.path.join(dataset_dir, f"signal_{num}.txt")
                output_path_filtered = os.path.join(dataset_dir, f"Python_filtered_signal_{num}.txt")
                np.savetxt(output_path_original, signal, fmt='%.6f')
                np.savetxt(output_path_filtered, filtered_signal, fmt='%.6f')
                print(f"信号已保存到: {output_path_original}")
                print(f"滤波后信号已保存到: {output_path_filtered}")
                
                plt.plot(signal[:1000])
                plt.title("Original Signal")
                plt.plot(filtered_signal[:1000])
                plt.title("Filtered Signal")
                plt.xlabel("Sample")
                plt.ylabel("Amplitude")
                plt.legend(["Original Signal", "Filtered Signal"])
                plt.show()
                # # 加载标注文件
                # annotation = wfdb.rdann(root + num, 'atr')
                # fs = annotation.fs
                # ann_len = annotation.ann_len
                # # MIT-BIH标注从1开始，需要转换为0-based索引
                # sig_sample = annotation.sample[1:]
                # # 创建QRS检测器实例
                # qrs_detector = PanTomkinsQRSDetectorOffline(signal_name=target_lead)
                # # 进行QRS检测
                # qrs_peaks = qrs_detector.detect_qrs_peaks(signal)
                # 
                # print(np.array(sig_sample))
                # print(np.array(sig_sample).shape)
                # print(np.array(qrs_peaks))
                # print(np.array(qrs_peaks).shape)

            except Exception as e:
                print(e)
    elif mode == 2:
        # 读取三个txt文件并绘制对比图
        input_path1 = "/home/yogsothoth/桌面/workspace-ppg/dataset/signal_100.txt"
        input_path2 = "/home/yogsothoth/桌面/workspace-ppg/dataset/Python_filtered_signal_100.txt"
        input_path3 = "/home/yogsothoth/桌面/workspace-ppg/dataset/CPP_filtered_signal_20Hz.txt"
        
        # 使用numpy读取文件，自动转换为float数组
        signal = np.loadtxt(input_path1)
        filtered_signal = np.loadtxt(input_path2)
        cpp_filtered_signal = np.loadtxt(input_path3)
        
        print(f"原始信号长度: {len(signal)}")
        print(f"Python滤波信号长度: {len(filtered_signal)}")
        print(f"C++滤波信号长度: {len(cpp_filtered_signal)}")
        
        # 创建一张图，显示三条曲线
        plt.figure(figsize=(12, 6))
        plt.plot(signal[:1000], label="Original Signal", alpha=0.7)
        plt.plot(filtered_signal[:1000], label="Python Filtered Signal", alpha=0.7)
        plt.plot(cpp_filtered_signal[:1000], label="C++ Filtered Signal", alpha=0.7)
        plt.title("Signal Comparison: Original vs Python Filter vs C++ Filter")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()