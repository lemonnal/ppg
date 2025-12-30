#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>
#include "DspFilters/Dsp.h"
#include "find_peaks.h"


// ===================== 零相位滤波核心函数 =====================

/**
 * @brief 零相位滤波函数（实现Python的filtfilt功能）
 * @param filter 滤波器对象
 * @param data 输入/输出数据（in-place修改）
 * @param numSamples 样本数量
 */
template<typename FilterType>
void filtfilt(FilterType& filter, float* data, int numSamples) {
    std::cout << "执行零相位滤波 (filtfilt)..." << std::endl;
    
    float* temp = new float[numSamples];
    float* temp_ptr = temp;  // 用于process函数的指针
    
    // 第一步：正向滤波
    memcpy(temp, data, numSamples * sizeof(float));
    filter.reset();
    filter.process(numSamples, &temp_ptr);
    std::cout << "  - 正向滤波完成" << std::endl;
    
    // 第二步：反转信号
    std::reverse(temp, temp + numSamples);
    std::cout << "  - 信号反转完成" << std::endl;
    
    // 第三步：反向滤波
    temp_ptr = temp;  // 重置指针
    filter.reset();
    filter.process(numSamples, &temp_ptr);
    std::cout << "  - 反向滤波完成" << std::endl;
    
    // 第四步：再次反转得到最终结果
    std::reverse(temp, temp + numSamples);
    std::cout << "  - 最终反转完成" << std::endl;
    
    // 复制结果到输出
    memcpy(data, temp, numSamples * sizeof(float));
    delete[] temp;
    
    std::cout << "零相位滤波完成！" << std::endl;
}

// ===================== 数据读取函数 =====================

/**
 * @brief 从文件读取信号数据到vector
 * @param filepath 输入文件路径
 * @param max_samples 最大读取样本数（0表示读取全部）
 * @return 信号数据数组
 */
std::vector<float> read_signal_from_file(const std::string& filepath, int max_samples = 0) {
    std::vector<float> signal;
    std::ifstream infile(filepath);
    
    if (!infile.is_open()) {
        std::cerr << "错误：无法打开文件 " << filepath << std::endl;
        return signal;
    }
    
    float value;
    int count = 0;
    
    while (infile >> value) {
        signal.push_back(value);
        count++;
        
        if (max_samples > 0 && count >= max_samples) {
            break;
        }
    }
    
    infile.close();
    std::cout << "从文件读取 " << signal.size() << " 个样本" << std::endl;
    
    return signal;
}

// ===================== 信号保存函数 =====================

/**
 * @brief 保存信号到文件（vector版本）
 * @param signal 信号数据
 * @param filepath 输出文件路径
 * @param precision 小数精度（默认6位）
 */
void save_signal_to_file(const std::vector<float>& signal, 
                         const std::string& filepath, int precision = 6) {
    std::ofstream outFile(filepath);
    
    if (!outFile.is_open()) {
        std::cerr << "错误：无法打开文件 " << filepath << std::endl;
        return;
    }
    
    for (float value : signal) {
        outFile << std::fixed << std::setprecision(precision) << value << std::endl;
    }
    
    outFile.close();
    std::cout << "信号已保存到: " << filepath << std::endl;
}

// ===================== 滤波函数（零相位）=====================

/**
 * @brief 零相位带通滤波（filtfilt）
 * @param input_signal 输入信号
 * @param low_freq 低频截止 (Hz)
 * @param high_freq 高频截止 (Hz)
 * @param sample_rate 采样率 (Hz)
 * @param filter_order 滤波器阶数
 * @return 滤波后的信号
 */
std::vector<float> apply_bandpass_zerophase(
    const std::vector<float>& input_signal,
    double low_freq,
    double high_freq,
    double sample_rate,
    int filter_order = 3
) {
    std::cout << "\n【零相位滤波】" << std::endl;
    std::cout << "  方法: filtfilt (正向+反向)" << std::endl;
    
    // 计算中心频率和带宽
    double center_frequency = std::sqrt(low_freq * high_freq);
    double bandwidth = high_freq - low_freq;
    
    // 创建滤波器
    Dsp::SimpleFilter<Dsp::Butterworth::BandPass<5>, 1> filter;
    filter.setup(filter_order, sample_rate, center_frequency, bandwidth);
    
    // 复制输入信号
    std::vector<float> output_signal = input_signal;
    
    // 应用filtfilt
    filtfilt(filter, output_signal.data(), output_signal.size());
    
    std::cout << "  零相位滤波完成！" << std::endl;
    
    return output_signal;
}

// ===================== 滤波函数（单向IIR）=====================

/**
 * @brief 单向IIR带通滤波（带均值初始化）
 * @param input_signal 输入信号
 * @param low_freq 低频截止 (Hz)
 * @param high_freq 高频截止 (Hz)
 * @param sample_rate 采样率 (Hz)
 * @param filter_order 滤波器阶数
 * @param use_warmup 是否使用均值初始化预热
 * @return 滤波后的信号
 */
std::vector<float> apply_bandpass_oneway(
    const std::vector<float>& input_signal,
    double low_freq,
    double high_freq,
    double sample_rate,
    int filter_order = 3,
    bool use_warmup = true
) {
    std::cout << "\n【单向IIR滤波】" << std::endl;
    std::cout << "  方法: 单向正向滤波" << std::endl;
    std::cout << "  预计群延迟: ~" << filter_order / (2 * M_PI * low_freq) * 1000 
              << " ms" << std::endl;
    
    // 计算中心频率和带宽
    double center_frequency = std::sqrt(low_freq * high_freq);
    double bandwidth = high_freq - low_freq;
    
    // 创建滤波器
    Dsp::SimpleFilter<Dsp::Butterworth::BandPass<5>, 1> filter;
    filter.setup(filter_order, sample_rate, center_frequency, bandwidth);
    
    // ========== 均值初始化预热 ==========
    if (use_warmup && input_signal.size() > 100) {
        // 计算前100个样本的均值
        int warmup_samples = std::min(100, (int)input_signal.size());
        float mean = 0.0f;
        for (int i = 0; i < warmup_samples; i++) {
            mean += input_signal[i];
        }
        mean /= warmup_samples;
        
        // 使用均值预热滤波器（约50次迭代）
        int warmup_iterations = 50;
        for (int i = 0; i < warmup_iterations; i++) {
            float temp = mean;
            float* p = &temp;
            filter.process(1, &p);
        }
        
        std::cout << "  滤波器预热: 是 (均值=" << std::fixed << std::setprecision(2) 
                  << mean << ", 迭代次数=" << warmup_iterations << ")" << std::endl;
    } else {
        std::cout << "  滤波器预热: 否" << std::endl;
    }
    
    // ========== 滤波处理 ==========
    std::vector<float> output_signal = input_signal;
    for (size_t i = 0; i < output_signal.size(); i++) {
        float* p = &output_signal[i];
        filter.process(1, &p);
    }
    
    std::cout << "  单向滤波完成！" << std::endl;
    
    return output_signal;
}

// ===================== 主函数 =====================

int main() {
    try {
        // PPG信号处理参数
        std::string input_file = "/home/yogsothoth/桌面/workspace-ppg/DataSet/PPG-BP/2_1.txt";
        std::string filtered_file_zerophase = "/home/yogsothoth/桌面/workspace-ppg/DSPFilter/PPG-BP/2_1_filtered_zerophase.txt";
        std::string filtered_file_oneway = "/home/yogsothoth/桌面/workspace-ppg/DSPFilter/PPG-BP/2_1_filtered_oneway.txt";
        
        // ========== 步骤1: 读取信号 ==========
        std::cout << "【步骤1: 读取信号】" << std::endl;
        std::vector<float> input_signal = read_signal_from_file(input_file, 2100);
        
        if (input_signal.empty()) {
            std::cerr << "错误：无法读取信号" << std::endl;
            return 1;
        }
        
        std::cout << "  信号长度: " << input_signal.size() << " 样本" << std::endl;
        std::cout << "  信号范围: [" << *std::min_element(input_signal.begin(), input_signal.end())
                  << ", " << *std::max_element(input_signal.begin(), input_signal.end()) << "]" << std::endl;
        
        // ========== 步骤2: 选择滤波方法 ==========
        int filter_method = 2;  // 1: 零相位滤波, 2: 单向IIR滤波
        
        std::vector<float> filtered_signal;
        std::string output_file;
        
        if (filter_method == 1) {
            std::cout << "\n【步骤2: 零相位滤波 (filtfilt)】" << std::endl;
            std::cout << "  优点: 无相位失真，波形保持好" << std::endl;
            std::cout << "  缺点: 需要完整信号，不适合实时\n" << std::endl;
            
            // 应用零相位滤波
            filtered_signal = apply_bandpass_zerophase(
                input_signal,    // 输入信号数组
                0.5,             // 低频截止 (Hz)
                20.0,            // 高频截止 (Hz)
                1000.0,          // 采样率 (Hz)
                3                // 滤波器阶数
            );
            
            output_file = filtered_file_zerophase;
        } 
        else if (filter_method == 2) {
            std::cout << "\n【步骤2: 单向IIR滤波】" << std::endl;
            std::cout << "  优点: 低延迟，逐样本处理，适合实时" << std::endl;
            std::cout << "  缺点: 有相位失真（群延迟）\n" << std::endl;
            
            // 应用单向IIR滤波
            filtered_signal = apply_bandpass_oneway(
                input_signal,    // 输入信号数组
                0.5,             // 低频截止 (Hz)
                20.0,            // 高频截止 (Hz)
                1000.0,          // 采样率 (Hz)
                3,               // 滤波器阶数
                true             // 使用均值初始化预热（减少瞬态响应）
            );
            
            output_file = filtered_file_oneway;
        }
        
        // ========== 步骤3: 保存结果 ==========
        std::cout << "\n【步骤3: 保存结果】" << std::endl;
        save_signal_to_file(filtered_signal, output_file);

        // ========== 步骤4: 峰值检测 ==========
        std::cout << "\n【步骤4: 峰值检测】" << std::endl;
        
        // 计算最小峰值间距 (PPG信号心率范围: 40-200 bpm)
        // 采样率1000Hz时，0.4秒 = 400样本，对应心率150 bpm
        double sample_rate = 1000.0;
        int min_distance = static_cast<int>(sample_rate * 0.4);
        
        std::cout << "  采样率: " << sample_rate << " Hz" << std::endl;
        std::cout << "  最小峰值间距: " << min_distance << " 样本 (" 
                  << (min_distance / sample_rate) << " 秒)" << std::endl;
        
        // 找峰值
        std::vector<int> peaks = find_peaks(filtered_signal, min_distance);
        std::cout << "  检测到峰值数量: " << peaks.size() << std::endl;
        
        // 找谷值（信号取反）
        std::vector<float> inverted_signal = filtered_signal;
        for (float& val : inverted_signal) {
            val = -val;
        }
        std::vector<int> valleys = find_peaks(inverted_signal, min_distance);
        std::cout << "  检测到谷值数量: " << valleys.size() << std::endl;
        
        // 打印前5个峰值
        if (!peaks.empty()) {
            std::cout << "\n  前" << std::min(5, (int)peaks.size()) << "个峰值:" << std::endl;
            for (int i = 0; i < std::min(5, (int)peaks.size()); i++) {
                int idx = peaks[i];
                std::cout << "    峰值 " << (i+1) << ": 位置=" << idx 
                          << " (" << std::fixed << std::setprecision(3) 
                          << (idx / sample_rate) << "s), 幅值=" 
                          << std::setprecision(2) << filtered_signal[idx] << std::endl;
            }
        }
        
        // 打印前5个谷值
        if (!valleys.empty()) {
            std::cout << "\n  前" << std::min(5, (int)valleys.size()) << "个谷值:" << std::endl;
            for (int i = 0; i < std::min(5, (int)valleys.size()); i++) {
                int idx = valleys[i];
                std::cout << "    谷值 " << (i+1) << ": 位置=" << idx 
                          << " (" << std::fixed << std::setprecision(3)
                          << (idx / sample_rate) << "s), 幅值=" 
                          << std::setprecision(2) << filtered_signal[idx] << std::endl;
            }
        }
        
        // 计算平均AC分量（峰峰值）
        if (!peaks.empty() && !valleys.empty()) {
            float sum_ac = 0;
            int count = 0;
            
            for (int peak_idx : peaks) {
                // 找最近的谷值
                int closest_valley = -1;
                int min_dist = std::numeric_limits<int>::max();
                
                for (int valley_idx : valleys) {
                    int dist = std::abs(valley_idx - peak_idx);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_valley = valley_idx;
                    }
                }
                
                if (closest_valley >= 0) {
                    float ac = std::abs(filtered_signal[peak_idx] - filtered_signal[closest_valley]);
                    sum_ac += ac;
                    count++;
                }
            }
            
            if (count > 0) {
                float avg_ac = sum_ac / count;
                std::cout << "\n  平均AC分量（峰峰值）: " << std::fixed 
                          << std::setprecision(2) << avg_ac << std::endl;
            }
        }
        
        // ========== 步骤5: 统计信息 ==========
        std::cout << "\n【步骤5: 统计信息】" << std::endl;
        
        float input_mean = 0, filtered_mean = 0;
        float input_energy = 0, filtered_energy = 0;
        
        for (size_t i = 0; i < input_signal.size(); i++) {
            input_mean += input_signal[i];
            input_energy += input_signal[i] * input_signal[i];
        }
        input_mean /= input_signal.size();
        input_energy /= input_signal.size();
        
        for (size_t i = 0; i < filtered_signal.size(); i++) {
            filtered_mean += filtered_signal[i];
            filtered_energy += filtered_signal[i] * filtered_signal[i];
        }
        filtered_mean /= filtered_signal.size();
        filtered_energy /= filtered_signal.size();
        
        std::cout << "  原始信号 - 均值: " << input_mean 
                  << ", 能量: " << input_energy << std::endl;
        std::cout << "  滤波信号 - 均值: " << filtered_mean 
                  << ", 能量: " << filtered_energy << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
