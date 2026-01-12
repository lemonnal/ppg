import numpy as np
from scipy import signal as scipy_signal
import wfdb
from signal_params import get_signal_params

class PanTomkinsQRSDetectorOffline:
    """
    基于Pan-Tomkins算法的QRS波检测器
    """

    def __init__(self, signal_name="MLII"):
        """
        初始化QRS检测器

        参数:
            fs: 采样频率 (Hz)
            signal_name: ECG导联名称 (如 "MLII", "V1", "V2" 等)
        """
        self.fs = 360
        self.signal = None
        self.filtered_signal = None
        self.differentiated_signal = None
        self.squared_signal = None
        self.integrated_signal = None
        self.qrs_peaks = []
        
        # 获取导联相关的处理参数并提取为直接属性（性能优化）
        params = get_signal_params('offline', signal_name)
        
        # 滤波参数
        self.low = params['low']
        self.high = params['high']
        self.filter_order = params['filter_order']
        self.original_weight = params['original_weight']
        self.filtered_weight = params['filtered_weight']
        
        # QRS检测参数
        self.integration_window_size = params['integration_window_size']
        self.detection_window_size = params['detection_window_size']
        self.overlap_window_size = params['overlap_window_size']
        self.refractory_period = int(params['refractory_period'] * self.fs)
        self.threshold_factor = params['threshold_factor']

    def bandpass_filter(self, signal_data):
        """
        自适应带通滤波器
        根据不同导联使用不同的频率参数

        参数:
            signal_data: 输入ECG信号

        返回:
            combined_signal: 滤波后与原始信号加权组合的信号
        """
        # 设计带通滤波器
        nyquist = 0.5 * self.fs
        low = self.low / nyquist
        high = self.high / nyquist

        # 使用 n 阶 Butterworth 滤波器 - 平衡滤波效果和信号保留
        b, a = scipy_signal.butter(self.filter_order, [low, high], btype='band')

        # 应用零相位滤波
        filtered_signal = scipy_signal.filtfilt(b, a, signal_data)

        # 添加原始信号的加权
        combined_signal = (self.original_weight * signal_data
                           + self.filtered_weight * filtered_signal)
        return combined_signal

    def derivative(self, signal_data):
        """
        优化的微分器 - 使用5点中心差分
        更好地突出QRS波的高斜率特性，减少噪声影响

        参数:
            signal_data: 输入信号

        返回:
            differentiated_signal: 微分后的信号
        """
        differentiated_signal = np.zeros_like(signal_data)

        # 使用5点中心差分公式提高精度
        # f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)
        for i in range(2, len(signal_data) - 2):
            differentiated_signal[i] = (-signal_data[i + 2] + 8 * signal_data[i + 1]
                                        - 8 * signal_data[i - 1] + signal_data[i - 2]) / 12

        return differentiated_signal

    def squaring(self, signal_data):
        """
        平方函数
        使所有点为正值，并放大高斜率点

        参数:
            signal_data: 输入信号

        返回:
            squared_signal: 平方后的信号
        """
        return signal_data ** 2

    def moving_window_integration(self, signal_data):
        """
        移动窗口积分器
        对微分平方后的信号进行平滑，突出QRS波特征

        参数:
            signal_data: 输入信号 (通常是微分平方后的信号)

        返回:
            integrated_signal: 移动平均积分后的信号
        """
        # 窗口中的采样点数量
        window_sample = int(self.integration_window_size * self.fs)

        # 使用卷积实现移动平均积分
        window = np.ones(window_sample) / window_sample
        integrated_signal = np.convolve(signal_data, window, mode='same')

        return integrated_signal

    def threshold_detection(self, signal_data):
        """
        滑动窗口阈值检测算法
        使用自适应的滑动窗口来适应信号变化，检测QRS波峰值

        参数:
            signal_data: 输入积分信号

        返回:
            refined_peaks: 定位的QRS波峰值位置列表
        """
        if signal_data is None or len(signal_data) == 0:
            return []

        # 设置滑动窗口参数
        window_size = int(self.detection_window_size * self.fs)  # 检测窗口大小 (秒)
        overlap_size = int(self.overlap_window_size * self.fs)    # 重叠窗口大小 (秒)

        # 使用预存储的实例属性（性能优化）
        refractory_period = self.refractory_period  # 不应期
        threshold_factor = self.threshold_factor    # 阈值系数

        all_peaks = [] # 检测到的R-peaks

        # 滑动窗口处理
        for start_idx in range(0, len(signal_data), overlap_size):
            end_idx = min(start_idx + window_size, len(signal_data))

            if end_idx - start_idx < overlap_size:  # 最后一个窗口太小就跳过
                break

            # 提取当前窗口的信号
            window_signal = signal_data[start_idx:end_idx]

            # 计算当前窗口的自适应阈值
            window_mean = np.mean(window_signal)
            window_std = np.std(window_signal)
            current_threshold = window_mean + threshold_factor * window_std

            # 在窗口内检测候选峰值
            window_peaks = []
            for i in range(len(window_signal)):
                actual_idx = start_idx + i
                current_value = window_signal[i]
                # 第一级过滤: 检查是否超过阈值
                if current_value > current_threshold:
                    # 第二级过滤: 检查是否在不应期内
                    if len(all_peaks) == 0 or (actual_idx - all_peaks[-1]) > refractory_period:
                        # 在窗口内寻找峰值点
                        search_range = min(10, len(window_signal) - i - 1)
                        local_peak_idx = i

                        for j in range(max(0, i - 5), min(len(window_signal), i + search_range + 1)):
                            if window_signal[j] > window_signal[local_peak_idx]:
                                local_peak_idx = j

                        # 添加找到的峰值 (避免重复)
                        if local_peak_idx not in window_peaks:
                            window_peaks.append(local_peak_idx)
                            all_peaks.append(start_idx + local_peak_idx)

        return all_peaks

    def detect_qrs_peaks(self, signal_data):
        """
        检测QRS波峰值

        参数:
            signal_data: 输入ECG信号

        返回:
            qrs_peaks: QRS波峰值位置索引
        """

        # 步骤1: 带通滤波
        self.filtered_signal = self.bandpass_filter(signal_data)

        # 步骤2: 微分
        self.differentiated_signal = self.derivative(self.filtered_signal)

        # 步骤3: 平方
        self.squared_signal = self.squaring(self.differentiated_signal)

        # 步骤4: 移动窗口积分
        self.integrated_signal = self.moving_window_integration(self.squared_signal)

        # 步骤5: QRS检测
        self.qrs_peaks = self.threshold_detection(self.integrated_signal)

        return self.qrs_peaks

    def calculate_rr_intervals(self, qrs_peaks=None):
        """
        计算RR间期（相邻R峰之间的时间间隔）
        
        参数:
            qrs_peaks: R峰位置列表（样本索引），如果为None则使用self.qrs_peaks
            
        返回:
            rr_intervals: RR间期列表（单位：秒）
            rr_intervals_samples: RR间期列表（单位：样本点）
        """
        if qrs_peaks is None:
            qrs_peaks = self.qrs_peaks
            
        if len(qrs_peaks) < 2:
            return [], []
            
        # 计算相邻R峰之间的样本数差值
        rr_intervals_samples = np.diff(qrs_peaks)
        
        # 转换为秒
        rr_intervals = rr_intervals_samples / self.fs
        
        return rr_intervals, rr_intervals_samples
    
    def calculate_instantaneous_heart_rate(self, qrs_peaks=None):
        """
        计算瞬时心率
        每个RR间期对应一个心率值
        
        参数:
            qrs_peaks: R峰位置列表（样本索引），如果为None则使用self.qrs_peaks
            
        返回:
            heart_rates: 瞬时心率列表（单位：bpm - 每分钟心跳次数）
            time_stamps: 对应的时间戳（单位：秒）
        """
        rr_intervals, _ = self.calculate_rr_intervals(qrs_peaks)
        
        if len(rr_intervals) == 0:
            return [], []
            
        # 心率(bpm) = 60 / RR间期(秒)
        heart_rates = 60.0 / rr_intervals
        
        # 时间戳对应每个RR间期的中点
        if qrs_peaks is None:
            qrs_peaks = self.qrs_peaks
        time_stamps = [(qrs_peaks[i] + qrs_peaks[i+1]) / 2.0 / self.fs 
                       for i in range(len(qrs_peaks) - 1)]
        
        return heart_rates, time_stamps
    
    def calculate_average_heart_rate(self, qrs_peaks=None):
        """
        计算平均心率
        基于所有RR间期的平均值
        
        参数:
            qrs_peaks: R峰位置列表（样本索引），如果为None则使用self.qrs_peaks
            
        返回:
            avg_heart_rate: 平均心率（单位：bpm）
            std_heart_rate: 心率标准差（单位：bpm）
        """
        heart_rates, _ = self.calculate_instantaneous_heart_rate(qrs_peaks)
        
        if len(heart_rates) == 0:
            return 0, 0
            
        avg_heart_rate = np.mean(heart_rates)
        std_heart_rate = np.std(heart_rates)
        
        return avg_heart_rate, std_heart_rate
    
    def calculate_sliding_window_heart_rate(self, window_size=10, qrs_peaks=None):
        """
        使用滑动窗口计算心率
        可以观察心率的时间变化趋势
        
        参数:
            window_size: 滑动窗口大小（RR间期的数量）
            qrs_peaks: R峰位置列表（样本索引），如果为None则使用self.qrs_peaks
            
        返回:
            sliding_hr: 滑动窗口心率列表（单位：bpm）
            time_stamps: 对应的时间戳（单位：秒）
        """
        heart_rates, hr_time_stamps = self.calculate_instantaneous_heart_rate(qrs_peaks)
        
        if len(heart_rates) < window_size:
            # 如果心率数据不足一个窗口，返回平均值
            if len(heart_rates) > 0:
                return [np.mean(heart_rates)], [hr_time_stamps[len(heart_rates)//2]]
            else:
                return [], []
        
        sliding_hr = []
        time_stamps = []
        
        # 滑动窗口计算
        for i in range(len(heart_rates) - window_size + 1):
            window_hr = heart_rates[i:i+window_size]
            avg_hr = np.mean(window_hr)
            sliding_hr.append(avg_hr)
            
            # 时间戳取窗口中点
            mid_idx = i + window_size // 2
            time_stamps.append(hr_time_stamps[mid_idx])
        
        return sliding_hr, time_stamps
    
    def get_heart_rate_statistics(self, qrs_peaks=None):
        """
        获取心率统计信息
        
        参数:
            qrs_peaks: R峰位置列表（样本索引），如果为None则使用self.qrs_peaks
            
        返回:
            stats: 包含各种心率统计信息的字典
        """
        heart_rates, _ = self.calculate_instantaneous_heart_rate(qrs_peaks)
        
        if len(heart_rates) == 0:
            return {
                'mean_hr': 0,
                'std_hr': 0,
                'min_hr': 0,
                'max_hr': 0,
                'median_hr': 0,
                'num_beats': 0
            }
        
        stats = {
            'mean_hr': np.mean(heart_rates),
            'std_hr': np.std(heart_rates),
            'min_hr': np.min(heart_rates),
            'max_hr': np.max(heart_rates),
            'median_hr': np.median(heart_rates),
            'num_beats': len(heart_rates) + 1  # RR间期数 + 1 = 心跳数
        }
        
        return stats

if __name__ == "__main__":
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

            print(f"\n处理记录 {num} - {target_lead} 导联")
            print(f"信号长度: {len(signal)} 样本")

            # 加载标注文件
            annotation = wfdb.rdann(root + num, 'atr')
            fs = annotation.fs
            ann_len = annotation.ann_len
            # MIT-BIH标注从1开始，需要转换为0-based索引
            sig_sample = annotation.sample[1:]
            # 创建QRS检测器实例
            qrs_detector = PanTomkinsQRSDetectorOffline(signal_name=target_lead)
            # 进行QRS检测
            qrs_peaks = qrs_detector.detect_qrs_peaks(signal)

            print(f"检测到 {len(qrs_peaks)} 个R峰")
            
            # ============ 心率计算示例 ============
            
            # 1. 计算RR间期
            rr_intervals, rr_samples = qrs_detector.calculate_rr_intervals()
            print(f"\nRR间期数量: {len(rr_intervals)}")
            if len(rr_intervals) > 0:
                print(f"RR间期范围: {np.min(rr_intervals):.3f}s - {np.max(rr_intervals):.3f}s")
                print(f"平均RR间期: {np.mean(rr_intervals):.3f}s")
            
            # 2. 计算瞬时心率
            inst_hr, hr_times = qrs_detector.calculate_instantaneous_heart_rate()
            print(f"\n瞬时心率数量: {len(inst_hr)}")
            if len(inst_hr) > 0:
                print(f"瞬时心率范围: {np.min(inst_hr):.1f} - {np.max(inst_hr):.1f} bpm")
            
            # 3. 计算平均心率
            avg_hr, std_hr = qrs_detector.calculate_average_heart_rate()
            print(f"\n平均心率: {avg_hr:.1f} ± {std_hr:.1f} bpm")
            
            # 4. 计算滑动窗口心率（窗口大小=10个心跳）
            sliding_hr, sliding_times = qrs_detector.calculate_sliding_window_heart_rate(window_size=10)
            if len(sliding_hr) > 0:
                print(f"\n滑动窗口心率（窗口=10）:")
                print(f"  数据点数: {len(sliding_hr)}")
                print(f"  心率范围: {np.min(sliding_hr):.1f} - {np.max(sliding_hr):.1f} bpm")
            
            # 5. 获取完整的心率统计信息
            stats = qrs_detector.get_heart_rate_statistics()
            print(f"\n=== 心率统计信息 ===")
            print(f"总心跳数: {stats['num_beats']}")
            print(f"平均心率: {stats['mean_hr']:.1f} bpm")
            print(f"标准差: {stats['std_hr']:.1f} bpm")
            print(f"最小心率: {stats['min_hr']:.1f} bpm")
            print(f"最大心率: {stats['max_hr']:.1f} bpm")
            print(f"中位数心率: {stats['median_hr']:.1f} bpm")
            
            # 显示前5个瞬时心率值作为示例
            if len(inst_hr) >= 5:
                print(f"\n前5个瞬时心率值:")
                for i in range(5):
                    print(f"  时间 {hr_times[i]:.2f}s: {inst_hr[i]:.1f} bpm")

        except Exception as e:
            print(f"错误: {e}")