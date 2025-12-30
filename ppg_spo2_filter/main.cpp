#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <vector>
#include <deque>
#include <algorithm>
#include "DspFilters/Dsp.h"

// ===================== 方案1：单向IIR滤波器 =====================

class RealtimeIIRFilter {
private:
    Dsp::SimpleFilter<Dsp::Butterworth::BandPass<5>, 1> filter;
    bool initialized;
    
public:
    RealtimeIIRFilter() : initialized(false) {}
    
    void setup(double sample_rate, double low_freq, double high_freq, int order = 5) {
        double center_freq = std::sqrt(low_freq * high_freq);
        double bandwidth = high_freq - low_freq;
        
        filter.setup(order, sample_rate, center_freq, bandwidth);
        filter.reset();
        initialized = true;
    }
    
    float processSample(float input) {
        if (!initialized) return input;
        
        float sample = input;
        float* p = &sample;
        filter.process(1, &p);
        return sample;
    }
};

// ===================== 方案2：移动平均+IIR滤波器 =====================

class RealtimeMovingAverageFilter {
private:
    Dsp::SimpleFilter<Dsp::Butterworth::BandPass<5>, 1> iir_filter;
    std::deque<float> buffer;
    int window_size;
    float sum;
    bool initialized;
    
public:
    RealtimeMovingAverageFilter() : window_size(5), sum(0.0f), initialized(false) {}
    
    void setup(double sample_rate, double low_freq, double high_freq, int ma_window = 5) {
        double center_freq = std::sqrt(low_freq * high_freq);
        double bandwidth = high_freq - low_freq;
        
        iir_filter.setup(5, sample_rate, center_freq, bandwidth);
        iir_filter.reset();
        
        window_size = ma_window;
        buffer.clear();
        sum = 0.0f;
        initialized = true;
    }
    
    float processSample(float input) {
        if (!initialized) return input;
        
        // 移动平均去除高频噪声
        buffer.push_back(input);
        sum += input;
        
        if (buffer.size() > static_cast<size_t>(window_size)) {
            sum -= buffer.front();
            buffer.pop_front();
        }
        
        float ma_output = sum / buffer.size();
        
        // IIR带通滤波
        float filtered = ma_output;
        float* p = &filtered;
        iir_filter.process(1, &p);
        
        return filtered;
    }
};

// ===================== 方案3：FIR滤波器 =====================

class RealtimeFIRFilter {
private:
    std::vector<float> fir_coeffs;
    std::deque<float> buffer;
    int fir_length;
    bool initialized;
    
public:
    RealtimeFIRFilter() : fir_length(101), initialized(false) {}
    
    void setup(double sample_rate, double low_freq, double high_freq, int fir_len = 101) {
        fir_length = fir_len | 1;  // 确保为奇数
        
        // 设计FIR带通滤波器（窗口法 + Hamming窗）
        fir_coeffs.resize(fir_length);
        int M = fir_length - 1;
        double w_low = 2.0 * M_PI * low_freq / sample_rate;
        double w_high = 2.0 * M_PI * high_freq / sample_rate;
        
        for (int n = 0; n < fir_length; ++n) {
            int k = n - M / 2;
            
            double h_ideal;
            if (k == 0) {
                h_ideal = (w_high - w_low) / M_PI;
            } else {
                h_ideal = (std::sin(w_high * k) - std::sin(w_low * k)) / (M_PI * k);
            }
            
            double window = 0.54 - 0.46 * std::cos(2.0 * M_PI * n / M);
            fir_coeffs[n] = h_ideal * window;
        }
        
        buffer.clear();
        initialized = true;
    }
    
    float processSample(float input) {
        if (!initialized) return input;
        
        buffer.push_back(input);
        
        if (buffer.size() > static_cast<size_t>(fir_length)) {
            buffer.pop_front();
        }
        
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

// ===================== 方案4：自适应滤波器 =====================

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
    
    void setup(double sample_rate, double low_freq, double high_freq) {
        base_filter.setup(sample_rate, low_freq, high_freq);
        variance_buffer.clear();
    }
    
    bool detectMotion(float current_sample) {
        variance_buffer.push_back(current_sample);
        
        if (variance_buffer.size() > static_cast<size_t>(variance_window)) {
            variance_buffer.pop_front();
        }
        
        if (variance_buffer.size() < static_cast<size_t>(variance_window)) {
            return false;
        }
        
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
        
        float std_dev = std::sqrt(variance);
        return std::abs(current_sample - mean) > threshold_multiplier * std_dev;
    }
    
    float processSample(float input) {
        motion_detected = detectMotion(input);
        
        float filtered = base_filter.processSample(input);
        
        if (motion_detected) {
            return 0.3f * input + 0.7f * filtered;  // 运动时
        } else {
            return 0.1f * input + 0.9f * filtered;  // 静止时
        }
    }
};

// ===================== 信号保存函数 =====================

void save_signal_to_file(const float* data, int numSamples, 
                         const std::string& filepath, int precision = 6) {
    std::ofstream outFile(filepath);
    
    if (!outFile.is_open()) {
        std::cerr << "错误：无法打开文件 " << filepath << std::endl;
        return;
    }
    
    for (int i = 0; i < numSamples; ++i) {
        outFile << std::fixed << std::setprecision(precision) << data[i] << std::endl;
    }
    
    outFile.close();
    std::cout << "信号已保存到: " << filepath << std::endl;
}

// ===================== 实时滤波处理函数 =====================

void realtime_bandpass_filter(
    const std::string& input_filepath,
    const std::string& output_filepath,
    double low_freq,
    double high_freq,
    double sample_rate,
    int method = 1,           // 1:IIR, 2:移动平均+IIR, 3:FIR, 4:自适应
    int filter_order = 5,
    float original_weight = 0.2f,
    float filtered_weight = 0.8f,
    int num_samples = 65000
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  实时带通滤波器处理" << std::endl;initialized) return input;
    std::cout << "========================================" << std::endl;
    
    // 显示参数
    std::cout << "\n【滤波器参数】" << std::endl;
    std::cout << "  低频截止: " << low_freq << " Hz" << std::endl;
    std::cout << "  高频截止: " << high_freq << " Hz" << std::endl;
    std::cout << "  采样率: " << sample_rate << " Hz" << std::endl;
    std::cout << "  加权系数: " << original_weight << "*原始 + " 
              << filtered_weight << "*滤波" << std::endl;
    
    // 计算中心频率和带宽
    double center_frequency = std::sqrt(low_freq * high_freq);
    double bandwidth = high_freq - low_freq;
    
    std::cout << "\n【计算结果】" << std::endl;
    std::cout << "  中心频率: " << center_frequency << " Hz" << std::endl;
    std::cout << "  带宽: " << bandwidth << " Hz" << std::endl;
    
    // 根据方法选择创建不同的滤波器
    std::cout << "\n【滤波器设置】" << std::endl;
    
    // 分配内存
    float* original = new float[num_samples];
    float* combined = new float[num_samples];
    
    // 读取原始信号
    std::cout << "\n【读取信号】" << std::endl;
    std::ifstream input_file(input_filepath);
    if (!input_file.is_open()) {
        std::cerr << "错误：无法打开输入文件 " << input_filepath << std::endl;
        delete[] original;
        delete[] combined;
        return;
    }
    
    for (int i = 0; i < num_samples; ++i) {
        input_file >> original[i];
    }
    input_file.close();
    std::cout << "  输入文件: " << input_filepath << std::endl;
    std::cout << "  样本数: " << num_samples << std::endl;
    
    // 实时滤波处理（根据选择的方法）
    std::cout << "\n【实时滤波处理】" << std::endl;
    
    if (method == 1) {
        std::cout << "  方法: 单向IIR滤波器" << std::endl;
        RealtimeIIRFilter filter;
        filter.setup(sample_rate, low_freq, high_freq, filter_order);
        
        for (int i = 0; i < num_samples; ++i) {
            float filtered = filter.processSample(original[i]);
            combined[i] = original_weight * original[i] + filtered_weight * filtered;
            
            if ((i + 1) % 10000 == 0) {
                std::cout << "  已处理: " << (i + 1) << " / " << num_samples 
                          << " (" << (i + 1) * 100 / num_samples << "%)" << std::endl;
            }
        }
    } 
    else if (method == 2) {
        std::cout << "  方法: 移动平均 + IIR滤波器" << std::endl;
        RealtimeMovingAverageFilter filter;
        filter.setup(sample_rate, low_freq, high_freq, 5);
        
        for (int i = 0; i < num_samples; ++i) {
            float filtered = filter.processSample(original[i]);
            combined[i] = original_weight * original[i] + filtered_weight * filtered;
            
            if ((i + 1) % 10000 == 0) {
                std::cout << "  已处理: " << (i + 1) << " / " << num_samples 
                          << " (" << (i + 1) * 100 / num_samples << "%)" << std::endl;
            }
        }
    }
    else if (method == 3) {
        std::cout << "  方法: FIR滤波器" << std::endl;
        RealtimeFIRFilter filter;
        filter.setup(sample_rate, low_freq, high_freq, 101);
        
        for (int i = 0; i < num_samples; ++i) {
            float filtered = filter.processSample(original[i]);
            combined[i] = original_weight * original[i] + filtered_weight * filtered;
            
            if ((i + 1) % 10000 == 0) {
                std::cout << "  已处理: " << (i + 1) << " / " << num_samples 
                          << " (" << (i + 1) * 100 / num_samples << "%)" << std::endl;
            }
        }
    }
    else if (method == 4) {
        std::cout << "  方法: 自适应滤波器" << std::endl;
        RealtimeAdaptiveFilter filter;
        filter.setup(sample_rate, low_freq, high_freq);
        
        for (int i = 0; i < num_samples; ++i) {
            float filtered = filter.processSample(original[i]);
            combined[i] = original_weight * original[i] + filtered_weight * filtered;
            
            if ((i + 1) % 10000 == 0) {
                std::cout << "  已处理: " << (i + 1) << " / " << num_samples 
                          << " (" << (i + 1) * 100 / num_samples << "%)" << std::endl;
            }
        }
    }
    
    std::cout << "实时滤波完成！" << std::endl;
    
    // 保存结果
    std::cout << "\n【保存结果】" << std::endl;
    save_signal_to_file(combined, num_samples, output_filepath);
    
    // 计算统计信息
    double energy = 0;
    for (int i = 0; i < num_samples; ++i) {
        energy += combined[i] * combined[i];
    }
    
    std::cout << "\n【统计信息】" << std::endl;
    std::cout << "  输出能量: " << energy / num_samples << std::endl;
    
    // 清理内存
    delete[] original;
    delete[] combined;
    
    std::cout << "\n处理完成！" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ===================== 主函数 =====================

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "  实时血氧滤波系统 - 4种方案" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        std::cout << "请选择滤波方法：" << std::endl;
        std::cout << "  1 - 单向IIR (延迟最低 ~50ms)" << std::endl;
        std::cout << "  2 - 移动平均+IIR (抗噪声强)" << std::endl;
        std::cout << "  3 - FIR滤波 (线性相位，延迟较大)" << std::endl;
        std::cout << "  4 - 自适应滤波 (运动检测)" << std::endl;
        std::cout << "\n输入选项 (1-4): ";
        
        int method;
        std::cin >> method;
        
        if (method < 1 || method > 4) {
            std::cerr << "无效选项，使用默认方法1" << std::endl;
            method = 1;
        }
        
        // 信号处理参数
        std::string input_file = "/home/yogsothoth/桌面/workspace-ppg/dataset/signal_100.txt";
        std::string output_file = "/home/yogsothoth/桌面/workspace-ppg/dataset/CPP_filtered_signal_1.txt";
        
        // 调用实时带通滤波处理
        realtime_bandpass_filter(
            input_file,           // 输入文件路径
            output_file,          // 输出文件路径
            5.0,                  // 低频截止 (Hz)
            20.0,                 // 高频截止 (Hz)
            360.0,                // 采样率 (Hz)
            method,               // 滤波方法 (1-4)
            5,                    // 滤波器阶数
            0.2f,                 // 原始信号权重
            0.8f,                 // 滤波信号权重
            65000                 // 样本数量
        );

    } catch (const std::exception& e) {
        std::cerr << "\n错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

