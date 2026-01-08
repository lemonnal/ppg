#include "include/realtime_filter.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

namespace ppg {

// ==================== RealtimeFilter 实现 ====================

RealtimeFilter::RealtimeFilter(double low_freq, double high_freq, 
                               double sample_rate, int filter_order)
    : low_freq_(low_freq), high_freq_(high_freq), 
      sample_rate_(sample_rate), filter_order_(filter_order) {
    
    // 计算中心频率和带宽
    double center_frequency = std::sqrt(low_freq_ * high_freq_);
    double bandwidth = high_freq_ - low_freq_;
    
    // 设置Butterworth带通滤波器
    filter_.setup(filter_order_, sample_rate_, center_frequency, bandwidth);
    
    std::cout << "实时滤波器初始化:" << std::endl;
    std::cout << "  - 低频截止: " << low_freq_ << " Hz" << std::endl;
    std::cout << "  - 高频截止: " << high_freq_ << " Hz" << std::endl;
    std::cout << "  - 中心频率: " << center_frequency << " Hz" << std::endl;
    std::cout << "  - 带宽: " << bandwidth << " Hz" << std::endl;
    std::cout << "  - 采样率: " << sample_rate_ << " Hz" << std::endl;
    std::cout << "  - 阶数: " << filter_order_ << std::endl;
}

float RealtimeFilter::process_sample(float input) {
    float* p = &input;
    filter_.process(1, &p);
    return input;
}

void RealtimeFilter::reset() {
    filter_.reset();
}

void RealtimeFilter::warmup(float initial_value, int num_samples) {
    std::cout << "滤波器预热中 (使用初始值: " << initial_value 
              << ", 预热样本数: " << num_samples << ")..." << std::endl;
    
    reset();
    
    // 用初始值喂入滤波器以建立稳定状态
    for (int i = 0; i < num_samples; ++i) {
        float temp = initial_value;
        float* p = &temp;
        filter_.process(1, &p);
    }
    
    std::cout << "滤波器预热完成！" << std::endl;
}

// ==================== RealtimeBuffer 实现 ====================

RealtimeBuffer::RealtimeBuffer(size_t capacity) 
    : capacity_(capacity) {
    buffer_.clear();
}

void RealtimeBuffer::push(float sample) {
    if (buffer_.size() >= capacity_) {
        buffer_.pop_front();  // 移除最旧的样本
    }
    buffer_.push_back(sample);  // 添加新样本
}

std::vector<float> RealtimeBuffer::get_data() const {
    return std::vector<float>(buffer_.begin(), buffer_.end());
}

// ==================== RealtimeBufferInt16 实现 ====================

RealtimeBufferInt16::RealtimeBufferInt16(size_t capacity) 
    : capacity_(capacity) {
    buffer_.clear();
}

void RealtimeBufferInt16::push(int16_t sample) {
    if (buffer_.size() >= capacity_) {
        buffer_.pop_front();  // 移除最旧的样本
    }
    buffer_.push_back(sample);  // 添加新样本
}

std::vector<int16_t> RealtimeBufferInt16::get_data_int() const {
    return std::vector<int16_t>(buffer_.begin(), buffer_.end());
}

std::vector<float> RealtimeBufferInt16::get_data_float() const {
    std::vector<float> result;
    result.reserve(buffer_.size());
    for (auto val : buffer_) {
        result.push_back(static_cast<float>(val));
    }
    return result;
}

std::vector<float> RealtimeBufferInt16::get_data_float(size_t start_idx, size_t length) const {
    if (start_idx >= buffer_.size()) {
        return {};
    }
    
    size_t actual_length = std::min(length, buffer_.size() - start_idx);
    std::vector<float> result;
    result.reserve(actual_length);
    
    auto it = buffer_.begin() + start_idx;
    for (size_t i = 0; i < actual_length; ++i, ++it) {
        result.push_back(static_cast<float>(*it));
    }
    
    return result;
}

} // namespace ppg

