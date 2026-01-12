#include "ecg_offline.hpp"
#include <DspFilters/Dsp.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace ecg {

// 构造函数
PanTomkinsQRSDetectorOffline::PanTomkinsQRSDetectorOffline(
    const std::string& signal_name, 
    float fs)
    : fs_(fs)
    , signal_name_(signal_name)
    , params_(get_signal_params(signal_name))
    , refractory_period_samples_(static_cast<int>(params_.refractory_period * fs_))
{
}

// 析构函数
PanTomkinsQRSDetectorOffline::~PanTomkinsQRSDetectorOffline() {
}

// 获取信号参数
SignalParams PanTomkinsQRSDetectorOffline::get_signal_params(const std::string& signal_name) {
    // 定义不同导联的参数配置
    static const std::map<std::string, SignalParams> params_map = {
        {"MLII", SignalParams()},  // 使用默认参数
        {"V1", []() {
            SignalParams p;
            p.low = 5.0f;
            p.high = 15.0f;
            p.filter_order = 4;
            p.original_weight = 0.2f;
            p.filtered_weight = 0.8f;
            p.integration_window_size = 0.150f;
            p.detection_window_size = 5.0f;
            p.overlap_window_size = 4.0f;
            p.refractory_period = 0.2f;
            p.threshold_factor = 0.8f;
            return p;
        }()},
        {"V2", []() {
            SignalParams p;
            p.low = 5.0f;
            p.high = 15.0f;
            p.filter_order = 4;
            p.original_weight = 0.2f;
            p.filtered_weight = 0.8f;
            p.integration_window_size = 0.150f;
            p.detection_window_size = 5.0f;
            p.overlap_window_size = 4.0f;
            p.refractory_period = 0.2f;
            p.threshold_factor = 0.8f;
            return p;
        }()}
    };
    
    auto it = params_map.find(signal_name);
    if (it != params_map.end()) {
        return it->second;
    }
    
    // 如果找不到，返回默认参数并给出警告
    std::cerr << "警告: 未找到导联 '" << signal_name 
              << "' 的参数配置，使用MLII默认参数" << std::endl;
    return SignalParams();
}

// 带通滤波器实现
std::vector<float> PanTomkinsQRSDetectorOffline::bandpass_filter(
    const std::vector<float>& signal_data) 
{
    if (signal_data.empty()) {
        return std::vector<float>();
    }
    
    // 计算中心频率和带宽
    double center_frequency = std::sqrt(params_.low * params_.high);
    double bandwidth = params_.high - params_.low;
    
    // 创建Butterworth带通滤波器
    Dsp::SimpleFilter<Dsp::Butterworth::BandPass<4>, 1> filter;
    filter.setup(params_.filter_order, fs_, center_frequency, bandwidth);
    
    // 复制信号用于滤波
    std::vector<float> filtered_signal = signal_data;
    
    // 前向滤波
    float* forward_ptr = filtered_signal.data();
    filter.reset();
    filter.process(static_cast<int>(filtered_signal.size()), &forward_ptr);
    
    // 反转信号
    std::reverse(filtered_signal.begin(), filtered_signal.end());
    
    // 反向滤波
    float* backward_ptr = filtered_signal.data();
    filter.reset();
    filter.process(static_cast<int>(filtered_signal.size()), &backward_ptr);
    
    // 再次反转得到最终滤波结果
    std::reverse(filtered_signal.begin(), filtered_signal.end());
    
    // 加权组合原始信号和滤波信号
    std::vector<float> combined_signal(signal_data.size());
    for (size_t i = 0; i < signal_data.size(); ++i) {
        combined_signal[i] = params_.original_weight * signal_data[i] 
                           + params_.filtered_weight * filtered_signal[i];
    }
    
    return combined_signal;
}

// 微分器实现 - 5点中心差分
std::vector<float> PanTomkinsQRSDetectorOffline::derivative(
    const std::vector<float>& signal_data)
{
    std::vector<float> differentiated_signal(signal_data.size(), 0.0f);
    
    // 使用5点中心差分公式
    // f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)
    for (size_t i = 2; i < signal_data.size() - 2; ++i) {
        differentiated_signal[i] = (-signal_data[i + 2] + 8.0f * signal_data[i + 1]
                                   - 8.0f * signal_data[i - 1] + signal_data[i - 2]) / 12.0f;
    }
    
    return differentiated_signal;
}

// 平方函数实现
std::vector<float> PanTomkinsQRSDetectorOffline::squaring(
    const std::vector<float>& signal_data)
{
    std::vector<float> squared_signal(signal_data.size());
    for (size_t i = 0; i < signal_data.size(); ++i) {
        squared_signal[i] = signal_data[i] * signal_data[i];
    }
    return squared_signal;
}

// 移动窗口积分器实现
std::vector<float> PanTomkinsQRSDetectorOffline::moving_window_integration(
    const std::vector<float>& signal_data)
{
    int window_samples = static_cast<int>(params_.integration_window_size * fs_);
    std::vector<float> integrated_signal(signal_data.size(), 0.0f);
    
    // 使用滑动窗口计算移动平均
    float sum = 0.0f;
    int half_window = window_samples / 2;
    
    for (size_t i = 0; i < signal_data.size(); ++i) {
        // 计算窗口边界
        int start = std::max(0, static_cast<int>(i) - half_window);
        int end = std::min(static_cast<int>(signal_data.size()), 
                          static_cast<int>(i) + half_window + 1);
        
        // 计算窗口内的平均值
        sum = 0.0f;
        for (int j = start; j < end; ++j) {
            sum += signal_data[j];
        }
        integrated_signal[i] = sum / (end - start);
    }
    
    return integrated_signal;
}

// 阈值检测实现
std::vector<int> PanTomkinsQRSDetectorOffline::threshold_detection(
    const std::vector<float>& signal_data)
{
    if (signal_data.empty()) {
        return std::vector<int>();
    }
    
    int window_size = static_cast<int>(params_.detection_window_size * fs_);
    int overlap_size = static_cast<int>(params_.overlap_window_size * fs_);
    
    std::vector<int> all_peaks;
    
    // 滑动窗口处理
    for (int start_idx = 0; start_idx < static_cast<int>(signal_data.size()); start_idx += overlap_size) {
        int end_idx = std::min(start_idx + window_size, static_cast<int>(signal_data.size()));
        
        if (end_idx - start_idx < overlap_size) {
            break;
        }
        
        // 提取当前窗口的信号
        std::vector<float> window_signal(signal_data.begin() + start_idx, 
                                        signal_data.begin() + end_idx);
        
        // 计算当前窗口的自适应阈值
        float window_mean = std::accumulate(window_signal.begin(), window_signal.end(), 0.0f) 
                          / window_signal.size();
        
        float window_variance = 0.0f;
        for (float val : window_signal) {
            float diff = val - window_mean;
            window_variance += diff * diff;
        }
        float window_std = std::sqrt(window_variance / window_signal.size());
        float current_threshold = window_mean + params_.threshold_factor * window_std;
        
        // 在窗口内检测候选峰值
        std::vector<int> window_peaks;
        for (size_t i = 0; i < window_signal.size(); ++i) {
            int actual_idx = start_idx + static_cast<int>(i);
            float current_value = window_signal[i];
            
            // 第一级过滤: 检查是否超过阈值
            if (current_value > current_threshold) {
                // 第二级过滤: 检查是否在不应期内
                if (all_peaks.empty() || 
                    (actual_idx - all_peaks.back()) > refractory_period_samples_) {
                    
                    // 在窗口内寻找峰值点
                    int search_range = std::min(10, static_cast<int>(window_signal.size()) - static_cast<int>(i) - 1);
                    int local_peak_idx = static_cast<int>(i);
                    
                    int search_start = std::max(0, static_cast<int>(i) - 5);
                    int search_end = std::min(static_cast<int>(window_signal.size()), 
                                            static_cast<int>(i) + search_range + 1);
                    
                    for (int j = search_start; j < search_end; ++j) {
                        if (window_signal[j] > window_signal[local_peak_idx]) {
                            local_peak_idx = j;
                        }
                    }
                    
                    // 添加找到的峰值 (避免重复)
                    if (std::find(window_peaks.begin(), window_peaks.end(), local_peak_idx) 
                        == window_peaks.end()) {
                        window_peaks.push_back(local_peak_idx);
                        all_peaks.push_back(start_idx + local_peak_idx);
                    }
                }
            }
        }
    }
    
    return all_peaks;
}

// 检测QRS波峰值
std::vector<int> PanTomkinsQRSDetectorOffline::detect_qrs_peaks(
    const std::vector<float>& signal_data)
{
    // 步骤1: 带通滤波
    filtered_signal_ = bandpass_filter(signal_data);
    
    // 步骤2: 微分
    differentiated_signal_ = derivative(filtered_signal_);
    
    // 步骤3: 平方
    squared_signal_ = squaring(differentiated_signal_);
    
    // 步骤4: 移动窗口积分
    integrated_signal_ = moving_window_integration(squared_signal_);
    
    // 步骤5: QRS检测
    qrs_peaks_ = threshold_detection(integrated_signal_);
    
    return qrs_peaks_;
}

// 计算RR间期
void PanTomkinsQRSDetectorOffline::calculate_rr_intervals(
    const std::vector<int>& qrs_peaks,
    std::vector<float>& rr_intervals_sec,
    std::vector<int>& rr_intervals_samples)
{
    const std::vector<int>& peaks = qrs_peaks.empty() ? qrs_peaks_ : qrs_peaks;
    
    if (peaks.size() < 2) {
        rr_intervals_sec.clear();
        rr_intervals_samples.clear();
        return;
    }
    
    rr_intervals_samples.resize(peaks.size() - 1);
    rr_intervals_sec.resize(peaks.size() - 1);
    
    for (size_t i = 0; i < peaks.size() - 1; ++i) {
        rr_intervals_samples[i] = peaks[i + 1] - peaks[i];
        rr_intervals_sec[i] = static_cast<float>(rr_intervals_samples[i]) / fs_;
    }
}

// 计算瞬时心率
void PanTomkinsQRSDetectorOffline::calculate_instantaneous_heart_rate(
    const std::vector<int>& qrs_peaks,
    std::vector<float>& heart_rates,
    std::vector<float>& time_stamps)
{
    std::vector<float> rr_intervals_sec;
    std::vector<int> rr_intervals_samples;
    calculate_rr_intervals(qrs_peaks, rr_intervals_sec, rr_intervals_samples);
    
    if (rr_intervals_sec.empty()) {
        heart_rates.clear();
        time_stamps.clear();
        return;
    }
    
    const std::vector<int>& peaks = qrs_peaks.empty() ? qrs_peaks_ : qrs_peaks;
    
    heart_rates.resize(rr_intervals_sec.size());
    time_stamps.resize(rr_intervals_sec.size());
    
    for (size_t i = 0; i < rr_intervals_sec.size(); ++i) {
        // 心率(bpm) = 60 / RR间期(秒)
        heart_rates[i] = 60.0f / rr_intervals_sec[i];
        
        // 时间戳对应每个RR间期的中点
        time_stamps[i] = (peaks[i] + peaks[i + 1]) / 2.0f / fs_;
    }
}

// 计算平均心率
void PanTomkinsQRSDetectorOffline::calculate_average_heart_rate(
    const std::vector<int>& qrs_peaks,
    float& avg_heart_rate,
    float& std_heart_rate)
{
    std::vector<float> heart_rates;
    std::vector<float> time_stamps;
    calculate_instantaneous_heart_rate(qrs_peaks, heart_rates, time_stamps);
    
    if (heart_rates.empty()) {
        avg_heart_rate = 0.0f;
        std_heart_rate = 0.0f;
        return;
    }
    
    // 计算平均值
    avg_heart_rate = std::accumulate(heart_rates.begin(), heart_rates.end(), 0.0f) 
                    / heart_rates.size();
    
    // 计算标准差
    float variance = 0.0f;
    for (float hr : heart_rates) {
        float diff = hr - avg_heart_rate;
        variance += diff * diff;
    }
    std_heart_rate = std::sqrt(variance / heart_rates.size());
}

// 计算滑动窗口心率
void PanTomkinsQRSDetectorOffline::calculate_sliding_window_heart_rate(
    int window_size,
    const std::vector<int>& qrs_peaks,
    std::vector<float>& sliding_hr,
    std::vector<float>& time_stamps)
{
    std::vector<float> heart_rates;
    std::vector<float> hr_time_stamps;
    calculate_instantaneous_heart_rate(qrs_peaks, heart_rates, hr_time_stamps);
    
    if (heart_rates.size() < static_cast<size_t>(window_size)) {
        // 如果心率数据不足一个窗口，返回平均值
        if (!heart_rates.empty()) {
            float avg = std::accumulate(heart_rates.begin(), heart_rates.end(), 0.0f) 
                       / heart_rates.size();
            sliding_hr.push_back(avg);
            time_stamps.push_back(hr_time_stamps[heart_rates.size() / 2]);
        } else {
            sliding_hr.clear();
            time_stamps.clear();
        }
        return;
    }
    
    sliding_hr.clear();
    time_stamps.clear();
    
    // 滑动窗口计算
    for (size_t i = 0; i <= heart_rates.size() - window_size; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < window_size; ++j) {
            sum += heart_rates[i + j];
        }
        float avg_hr = sum / window_size;
        sliding_hr.push_back(avg_hr);
        
        // 时间戳取窗口中点
        int mid_idx = static_cast<int>(i) + window_size / 2;
        time_stamps.push_back(hr_time_stamps[mid_idx]);
    }
}

// 获取心率统计信息
HeartRateStats PanTomkinsQRSDetectorOffline::get_heart_rate_statistics(
    const std::vector<int>& qrs_peaks)
{
    HeartRateStats stats;
    std::vector<float> heart_rates;
    std::vector<float> time_stamps;
    calculate_instantaneous_heart_rate(qrs_peaks, heart_rates, time_stamps);
    
    if (heart_rates.empty()) {
        stats.mean_hr = 0.0f;
        stats.std_hr = 0.0f;
        stats.min_hr = 0.0f;
        stats.max_hr = 0.0f;
        stats.median_hr = 0.0f;
        stats.num_beats = 0;
        return stats;
    }
    
    // 平均值
    stats.mean_hr = std::accumulate(heart_rates.begin(), heart_rates.end(), 0.0f) 
                   / heart_rates.size();
    
    // 标准差
    float variance = 0.0f;
    for (float hr : heart_rates) {
        float diff = hr - stats.mean_hr;
        variance += diff * diff;
    }
    stats.std_hr = std::sqrt(variance / heart_rates.size());
    
    // 最小值和最大值
    stats.min_hr = *std::min_element(heart_rates.begin(), heart_rates.end());
    stats.max_hr = *std::max_element(heart_rates.begin(), heart_rates.end());
    
    // 中位数
    std::vector<float> sorted_hr = heart_rates;
    std::sort(sorted_hr.begin(), sorted_hr.end());
    if (sorted_hr.size() % 2 == 0) {
        stats.median_hr = (sorted_hr[sorted_hr.size() / 2 - 1] 
                         + sorted_hr[sorted_hr.size() / 2]) / 2.0f;
    } else {
        stats.median_hr = sorted_hr[sorted_hr.size() / 2];
    }
    
    // 心跳数 = RR间期数 + 1
    stats.num_beats = static_cast<int>(heart_rates.size()) + 1;
    
    return stats;
}

// 从文件读取ECG信号
std::vector<float> read_ecg_signal(const std::string& filename, size_t max_samples) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename);
    }
    
    std::vector<float> signal;
    float value;
    size_t count = 0;
    
    while (file >> value) {
        signal.push_back(value);
        ++count;
        if (max_samples > 0 && count >= max_samples) {
            break;
        }
    }
    
    file.close();
    return signal;
}

// 保存信号到文件
bool save_signal_to_file(const std::vector<float>& signal, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }
    
    for (float value : signal) {
        file << value << "\n";
    }
    
    file.close();
    return true;
}

} // namespace ecg

