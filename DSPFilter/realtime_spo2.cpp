#include <iostream>
#include <cmath>
#include <vector>
#include <deque>
#include <chrono>
#include "DspFilters/Dsp.h"

// =====================================================================
// 方案1：单向IIR滤波器（最低延迟，适合实时心率监测）
// =====================================================================

class RealtimeIIRFilter {
private:
    Dsp::SimpleFilter<Dsp::Butterworth::BandPass<5>, 1> filter;
    bool initialized;
    
public:
    RealtimeIIRFilter() : initialized(false) {}
    
    /**
     * @brief 初始化滤波器
     * @param sample_rate 采样率 (Hz)
     * @param low_freq 低频截止 (Hz)
     * @param high_freq 高频截止 (Hz)
     * @param order 滤波器阶数
     */
    void setup(double sample_rate, double low_freq, double high_freq, int order = 5) {
        double center_freq = std::sqrt(low_freq * high_freq);
        double bandwidth = high_freq - low_freq;
        
        filter.setup(order, sample_rate, center_freq, bandwidth);
        filter.reset();
        initialized = true;
        
        std::cout << "【方案1: 单向IIR滤波器】" << std::endl;
        std::cout << "  延迟: ~" << order / (2.0 * low_freq) * 1000 << " ms" << std::endl;
        std::cout << "  相位失真: 有（线性相位）" << std::endl;
        std::cout << "  计算量: 最低" << std::endl;
    }
    
    /**
     * @brief 处理单个样本（流式处理）
     * @param input 输入样本
     * @return 滤波后的样本
     */
    float processSample(float input) {
        if (!initialized) {
            std::cerr << "错误：滤波器未初始化！" << std::endl;
            return input;
        }
        
        float sample = input;
        filter.process(1, &sample);  // 处理单个样本
        return sample;
    }
    
    /**
     * @brief 批量处理（用于测试）
     * @param data 输入数据数组
     * @param output 输出数据数组
     * @param num_samples 样本数
     */
    void processBatch(const float* data, float* output, int num_samples) {
        for (int i = 0; i < num_samples; ++i) {
            output[i] = processSample(data[i]);
        }
    }
};

// =====================================================================
// 方案2：移动平均预滤波 + IIR滤波（抗噪声能力强）
// =====================================================================

class RealtimeMovingAverageFilter {
private:
    Dsp::SimpleFilter<Dsp::Butterworth::BandPass<5>, 1> iir_filter;
    std::deque<float> buffer;
    int window_size;
    float sum;
    bool initialized;
    
public:
    RealtimeMovingAverageFilter() : window_size(5), sum(0.0f), initialized(false) {}
    
    /**
     * @brief 初始化滤波器
     * @param sample_rate 采样率 (Hz)
     * @param low_freq 低频截止 (Hz)
     * @param high_freq 高频截止 (Hz)
        # 创建一张图，显示四条曲线
     * @param ma_window 移动平均窗口大小（建议3-10）
     */
    void setup(double sample_rate, double low_freq, double high_freq, int ma_window = 5) {
        double center_freq = std::sqrt(low_freq * high_freq);
        double bandwidth = high_freq - low_freq;
        
        iir_filter.setup(5, sample_rate, center_freq, bandwidth);
        iir_filter.reset();
        
        window_size = ma_window;
        buffer.clear();
        sum = 0.0f;
        initialized = true;
        
        std::cout << "【方案2: 移动平均 + IIR滤波器】" << std::endl;
        std::cout << "  延迟: ~" << (window_size / 2.0) / sample_rate * 1000 + 50 << " ms" << std::endl;
        std::cout << "  抗噪声: 强" << std::endl;
        std::cout << "  计算量: 中等" << std::endl;
    }
    
    /**
     * @brief 处理单个样本
     */
    float processSample(float input) {
        if (!initialized) return input;
        
        // 第一步：移动平均去除高频噪声
        buffer.push_back(input);
        sum += input;
        
        if (buffer.size() > static_cast<size_t>(window_size)) {
            sum -= buffer.front();
            buffer.pop_front();
        }
        
        float ma_output = sum / buffer.size();
        
        // 第二步：IIR带通滤波
        float filtered = ma_output;
        iir_filter.process(1, &filtered);
        
        return filtered;
    }
};

// =====================================================================
// 方案3：环形缓冲区 + 分块FIR滤波（零相位，但有延迟）
// =====================================================================

class RealtimeBlockFIRFilter {
private:
    std::vector<float> fir_coeffs;
    std::deque<float> buffer;
    int block_size;
    int fir_length;
    bool initialized;
    
public:
    RealtimeBlockFIRFilter() : block_size(32), fir_length(101), initialized(false) {}
    
    /**
     * @brief 初始化FIR滤波器（使用窗口法设计）
     * @param sample_rate 采样率 (Hz)
     * @param low_freq 低频截止 (Hz)
     * @param high_freq 高频截止 (Hz)
     * @param fir_len FIR滤波器长度（必须为奇数）
     * @param blk_size 分块大小
     */
    void setup(double sample_rate, double low_freq, double high_freq, 
               int fir_len = 101, int blk_size = 32) {
        fir_length = fir_len | 1;  // 确保为奇数
        block_size = blk_size;
        
        // 使用窗口法设计FIR带通滤波器
        designBandpassFIR(sample_rate, low_freq, high_freq);
        
        buffer.clear();
        initialized = true;
        
        std::cout << "【方案3: 分块FIR滤波器】" << std::endl;
        std::cout << "  延迟: ~" << (fir_length / 2.0) / sample_rate * 1000 << " ms" << std::endl;
        std::cout << "  相位失真: 无（线性相位）" << std::endl;
        std::cout << "  计算量: 高" << std::endl;
        std::cout << "  FIR长度: " << fir_length << std::endl;
    }
    
    /**
     * @brief 设计带通FIR滤波器（窗口法 + Hamming窗）
     */
    void designBandpassFIR(double fs, double f_low, double f_high) {
        fir_coeffs.resize(fir_length);
        int M = fir_length - 1;
        double w_low = 2.0 * M_PI * f_low / fs;
        double w_high = 2.0 * M_PI * f_high / fs;
        
        for (int n = 0; n < fir_length; ++n) {
            int k = n - M / 2;
            
            // 理想带通滤波器的冲激响应
            double h_ideal;
            if (k == 0) {
                h_ideal = (w_high - w_low) / M_PI;
            } else {
                h_ideal = (std::sin(w_high * k) - std::sin(w_low * k)) / (M_PI * k);
            }
            
            // Hamming窗
            double window = 0.54 - 0.46 * std::cos(2.0 * M_PI * n / M);
            
            fir_coeffs[n] = h_ideal * window;
        }
    }
    
    /**
     * @brief 处理单个样本（使用环形缓冲区）
     */
    float processSample(float input) {
        if (!initialized) return input;
        
        buffer.push_back(input);
        
        // 保持缓冲区大小
        if (buffer.size() > static_cast<size_t>(fir_length)) {
            buffer.pop_front();
        }
        
        // 如果缓冲区未满，返回原值（初始化阶段）
        if (buffer.size() < static_cast<size_t>(fir_length)) {
            return input;
        }
        
        // FIR卷积
        float output = 0.0f;
        for (int i = 0; i < fir_length; ++i) {
            output += fir_coeffs[i] * buffer[i];
        }
        
        return output;
    }
};

// =====================================================================
// 方案4：自适应滤波器（最智能，适合运动伪影抑制）
// =====================================================================

class RealtimeAdaptiveFilter {
private:
    RealtimeIIRFilter base_filter;
    std::deque<float> variance_buffer;
    int variance_window;
    float threshold_multiplier;
    bool motion_detected;
    
public:
    RealtimeAdaptiveFilter() : variance_window(50), threshold_multiplier(2.0f), 
                                motion_detected(false) {}
    
    /**
     * @brief 初始化自适应滤波器
     */
    void setup(double sample_rate, double low_freq, double high_freq) {
        base_filter.setup(sample_rate, low_freq, high_freq);
        variance_buffer.clear();
        
        std::cout << "【方案4: 自适应滤波器】" << std::endl;
        std::cout << "  特点: 运动检测 + 自适应参数调整" << std::endl;
        std::cout << "  适用: 运动伪影抑制" << std::endl;
    }
    
    /**
     * @brief 检测运动伪影（通过方差突变）
     */
    bool detectMotion(float current_sample) {
        variance_buffer.push_back(current_sample);
        
        if (variance_buffer.size() > static_cast<size_t>(variance_window)) {
            variance_buffer.pop_front();
        }
        
        if (variance_buffer.size() < static_cast<size_t>(variance_window)) {
            return false;
        }
        
        // 计算均值和方差
        float mean = 0.0f;
        for (float val : variance_buffer) {
            mean += val;
        }
        mean /= variance_buffer.size();
        
        float variance = 0.0f;
        for (float val : variance_buffer) {
            variance += (val - mean) * (val - mean);
        }
        variance /= variance_buffer.size();
        
        // 简单的阈值检测
        float std_dev = std::sqrt(variance);
        return std::abs(current_sample - mean) > threshold_multiplier * std_dev;
    }
    
    /**
     * @brief 处理单个样本（运动时增强平滑）
     */
    float processSample(float input) {
        motion_detected = detectMotion(input);
        
        float filtered = base_filter.processSample(input);
        
        // 运动时增加平滑（降低权重）
        if (motion_detected) {
            return 0.3f * input + 0.7f * filtered;  // 运动时
        } else {
            return 0.1f * input + 0.9f * filtered;  // 静止时
        }
    }
    
    bool isMotionDetected() const { return motion_detected; }
};

// =====================================================================
// 完整的实时血氧处理系统
// =====================================================================

class RealtimeSPO2System {
private:
    RealtimeIIRFilter red_filter;     // 红光通道
    RealtimeIIRFilter ir_filter;      // 红外通道
    
    std::deque<float> red_buffer;
    std::deque<float> ir_buffer;
    int peak_detection_window;
    
    // 血氧计算参数
    float R_value;  // R = (AC_red/DC_red) / (AC_ir/DC_ir)
    
public:
    RealtimeSPO2System() : peak_detection_window(100), R_value(0.0f) {}
    
    /**
     * @brief 初始化双通道血氧系统
     */
    void setup(double sample_rate = 100.0) {
        // 心率频带：0.5-4.0 Hz (30-240 bpm)
        red_filter.setup(sample_rate, 0.5, 4.0, 4);
        ir_filter.setup(sample_rate, 0.5, 4.0, 4);
        
        red_buffer.clear();
        ir_buffer.clear();
        
        std::cout << "\n【完整实时血氧系统】" << std::endl;
        std::cout << "  双通道: 红光(660nm) + 红外(940nm)" << std::endl;
        std::cout << "  采样率: " << sample_rate << " Hz" << std::endl;
        std::cout << "  输出: SpO2%, HR(bpm)" << std::endl;
    }
    
    /**
     * @brief 处理一对红光/红外样本
     * @param red_sample 红光ADC值
     * @param ir_sample 红外ADC值
     * @param spo2 输出血氧饱和度(%)
     * @param heart_rate 输出心率(bpm)
     */
    void processSample(float red_sample, float ir_sample, 
                       float& spo2, float& heart_rate) {
        // 滤波处理
        float red_filtered = red_filter.processSample(red_sample);
        float ir_filtered = ir_filter.processSample(ir_sample);
        
        // 存入缓冲区用于AC/DC计算
        red_buffer.push_back(red_filtered);
        ir_buffer.push_back(ir_filtered);
        
        if (red_buffer.size() > static_cast<size_t>(peak_detection_window)) {
            red_buffer.pop_front();
            ir_buffer.pop_front();
        }
        
        // 需要足够数据才能计算
        if (red_buffer.size() < static_cast<size_t>(peak_detection_window)) {
            spo2 = 0.0f;
            heart_rate = 0.0f;
            return;
        }
        
        // 计算AC和DC分量
        float red_ac, red_dc, ir_ac, ir_dc;
        calculateACDC(red_buffer, red_ac, red_dc);
        calculateACDC(ir_buffer, ir_ac, ir_dc);
        
        // 计算R值
        if (ir_dc > 0.01f && red_dc > 0.01f) {
            R_value = (red_ac / red_dc) / (ir_ac / ir_dc);
        }
        
        // 根据R值计算SpO2（经验公式）
        spo2 = 110.0f - 25.0f * R_value;
        
        // 限制范围
        if (spo2 > 100.0f) spo2 = 100.0f;
        if (spo2 < 70.0f) spo2 = 70.0f;
        # 创建一张图，显示四条曲线
        
        // 计算心率（通过峰值检测）
        heart_rate = calculateHeartRate(ir_buffer, 100.0);
    }
    
private:
    /**
     * @brief 计算AC和DC分量
     */
    void calculateACDC(const std::deque<float>& buffer, float& ac, float& dc) {
        float max_val = *std::max_element(buffer.begin(), buffer.end());
        float min_val = *std::min_element(buffer.begin(), buffer.end());
        
        ac = (max_val - min_val) / 2.0f;  // 峰峰值的一半
        dc = (max_val + min_val) / 2.0f;  // 均值
    }
    
    /**
     * @brief 简单的心率计算（峰值检测）
     */
    float calculateHeartRate(const std::deque<float>& buffer, float sample_rate) {
        int peak_count = 0;
        float threshold = 0.0f;
        
        // 计算阈值（均值）
        for (float val : buffer) {
            threshold += val;
        }
        threshold /= buffer.size();
        
        // 峰值检测
        bool above_threshold = false;
        for (size_t i = 1; i < buffer.size() - 1; ++i) {
            if (buffer[i] > threshold && buffer[i] > buffer[i-1] && 
                buffer[i] > buffer[i+1]) {
                if (!above_threshold) {
                    peak_count++;
                    above_threshold = true;
                }
            } else if (buffer[i] < threshold) {
                above_threshold = false;
            }
        }
        
        // 转换为bpm
        float duration_sec = buffer.size() / sample_rate;
        return (peak_count / duration_sec) * 60.0f;
    }
};

// =====================================================================
// 性能测试函数
// =====================================================================

void performanceTest() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  实时性能测试" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    const int TEST_SAMPLES = 10000;
    const double SAMPLE_RATE = 100.0;
    
    // 生成测试信号（模拟心率）
    std::vector<float> test_signal(TEST_SAMPLES);
    for (int i = 0; i < TEST_SAMPLES; ++i) {
        float t = i / SAMPLE_RATE;
        test_signal[i] = std::sin(2.0 * M_PI * 1.2 * t) +  // 1.2 Hz心率
                        0.3f * std::sin(2.0 * M_PI * 10.0 * t); // 高频噪声
    }
    
    // 测试方案1
    {
        RealtimeIIRFilter filter;
        filter.setup(SAMPLE_RATE, 0.5, 4.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < TEST_SAMPLES; ++i) {
            filter.processSample(test_signal[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time = duration.count() / (double)TEST_SAMPLES;
        double max_rate = 1e6 / avg_time;
        
        std::cout << "方案1 单向IIR: " << avg_time << " μs/样本, 最大采样率: " 
                  << max_rate << " Hz" << std::endl;
    }
    
    // 测试方案2
    {
        RealtimeMovingAverageFilter filter;
        filter.setup(SAMPLE_RATE, 0.5, 4.0, 5);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < TEST_SAMPLES; ++i) {
            filter.processSample(test_signal[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time = duration.count() / (double)TEST_SAMPLES;
        double max_rate = 1e6 / avg_time;
        
        std::cout << "方案2 移动平均+IIR: " << avg_time << " μs/样本, 最大采样率: " 
                  << max_rate << " Hz" << std::endl;
    }
    
    // 测试方案3
    {
        RealtimeBlockFIRFilter filter;
        filter.setup(SAMPLE_RATE, 0.5, 4.0, 51, 32);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < TEST_SAMPLES; ++i) {
            filter.processSample(test_signal[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time = duration.count() / (double)TEST_SAMPLES;
        double max_rate = 1e6 / avg_time;
        
        std::cout << "方案3 FIR滤波: " << avg_time << " μs/样本, 最大采样率: " 
                  << max_rate << " Hz" << std::endl;
    }
}

// =====================================================================
// 主函数：演示不同方案
// =====================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  实时血氧系统设计方案" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // 方案对比
    std::cout << "【方案对比】\n" << std::endl;
    std::cout << "方案1 - 单向IIR:\n"
              << "  优点: 延迟最低(~50ms), 计算量最小\n"
              << "  缺点: 有相位失真\n"
              << "  适用: 普通心率监测\n" << std::endl;
    
    std::cout << "方案2 - 移动平均+IIR:\n"
              << "  优点: 抗噪声能力强, 延迟适中(~80ms)\n"
              << "  缺点: 计算量稍大\n"
              << "  适用: 嘈杂环境\n" << std::endl;
    
    std::cout << "方案3 - 分块FIR:\n"
              << "  优点: 线性相位, 无失真\n"
              << "  缺点: 延迟较大(~140ms), 计算量大\n"
              << "  适用: 高精度应用\n" << std::endl;
    
    std::cout << "方案4 - 自适应滤波:\n"
              << "  优点: 智能运动检测, 参数自适应\n"
              << "  缺点: 复杂度最高\n"
              << "  适用: 运动场景\n" << std::endl;
    
    // 性能测试
    performanceTest();
    
    // 演示完整血氧系统
    std::cout << "\n========================================" << std::endl;
    std::cout << "  血氧系统实时演示" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    RealtimeSPO2System spo2_system;
    spo2_system.setup(100.0);  // 100 Hz采样率
    
    // 模拟采集10秒数据
    std::cout << "\n模拟实时采集（前20个样本）:\n" << std::endl;
    std::cout << "样本#  |  红光ADC  |  红外ADC  |  SpO2%  |  心率(bpm)" << std::endl;
    std::cout << "-------|-----------|-----------|---------|------------" << std::endl;
    
    for (int i = 0; i < 20; ++i) {
        float t = i / 100.0f;
        
        // 模拟PPG信号
        float red_adc = 2048 + 200 * std::sin(2.0 * M_PI * 1.2 * t) + 
                       50 * (rand() % 100 - 50) / 50.0f;
        float ir_adc = 2048 + 300 * std::sin(2.0 * M_PI * 1.2 * t) + 
                      50 * (rand() % 100 - 50) / 50.0f;
        
        float spo2, heart_rate;
        spo2_system.processSample(red_adc, ir_adc, spo2, heart_rate);
        
        printf("%6d | %9.1f | %9.1f | %6.1f%% | %6.1f\n", 
               i, red_adc, ir_adc, spo2, heart_rate);
    }
    
    std::cout << "\n注: 前100个样本为初始化阶段，之后数据趋于稳定" << std::endl;
    
    return 0;
}

