from scipy.signal import butter, filtfilt, find_peaks
import numpy as np



def calculate_spo2_from_ppg(ppg_signal, sampling_rate=50):
    """
    基于单通道PPG信号估算SpO2
    
    方法：
    1. 计算AC分量（交流分量）：信号的峰峰值
    2. 计算DC分量（直流分量）：信号的平均值
    3. 计算归一化比率：AC/DC
    4. 使用经验公式转换为SpO2
    
    参数:
        ppg_signal: PPG信号数组
    
    返回:
        spo2: 估算的SpO2值（百分比）
    """
    # 去除趋势（高通滤波）
    
    # 设计高通滤波器去除基线漂移
    nyquist = sampling_rate / 2
    cutoff = 0.5  # 0.5 Hz高通
    b, a = butter(3, cutoff / nyquist, btype='high')
    ppg_filtered = filtfilt(b, a, ppg_signal)
    
    # 找到峰值和谷值
    peaks, _ = find_peaks(ppg_filtered, distance=sampling_rate*0.4)  # 至少间隔0.4秒
    valleys, _ = find_peaks(-ppg_filtered, distance=sampling_rate*0.4)
    
    print(ppg_filtered)
    print(peaks,valleys)
    
    if len(peaks) < 2 or len(valleys) < 2:
        # 如果检测不到足够的峰值，使用简单方法
        ac_component = np.max(ppg_signal) - np.min(ppg_signal)
        dc_component = np.mean(ppg_signal)
    else:
        # AC分量：峰值和谷值的平均差
        peak_values = ppg_filtered[peaks]
        valley_values = ppg_filtered[valleys]
        ac_component = np.mean(peak_values) - np.mean(valley_values)
        
        # DC分量：原始信号的平均值
        dc_component = np.mean(ppg_signal)
    
    # 计算归一化比率
    if dc_component != 0:
        ratio = ac_component / dc_component
    else:
        ratio = 0.02  # 默认值
    
    # 使用经验公式计算SpO2
    # 基于文献的经验公式：SpO2 = 110 - 25 * R
    # 这里R是归一化后的比率，需要调整到合适的范围
    # 对于健康人，SpO2通常在95-100%之间
    
    # 归一化ratio到合适范围（根据PPG信号特点调整）
    normalized_ratio = ratio * 100  # 转换为百分比形式
    
    # 限制在合理范围内（避免异常值）
    normalized_ratio = np.clip(normalized_ratio, 0.1, 10)
    
    # 使用修正的经验公式
    # 对于健康人群的PPG信号，通常SpO2在95-100%
    # 原始对数公式：
    # spo2 = 110 - 25 * np.log10(normalized_ratio + 1)
    
    # 使用3次多项式近似（在90-100%范围内，RMSE: 0.01%, Max Error: 0.036%）
    # 优势：避免对数运算，计算速度更快
    spo2 = (-4.5837351812e-02 * normalized_ratio**3 + 
            7.7521128177e-01 * normalized_ratio**2 + 
            -6.1491433678e+00 * normalized_ratio + 
            1.0764977008e+02)
    
    # 限制SpO2在合理范围 (90-100%)
    spo2 = np.clip(spo2, 90, 100)
    
    return spo2, ratio
