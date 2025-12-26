#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <algorithm>
#include "DspFilters/Dsp.h"

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
    
    // 第一步：正向滤波
    memcpy(temp, data, numSamples * sizeof(float));
    filter.reset();
    filter.process(numSamples, &temp);
    std::cout << "  - 正向滤波完成" << std::endl;
    
    // 第二步：反转信号
    std::reverse(temp, temp + numSamples);
    std::cout << "  - 信号反转完成" << std::endl;
    
    // 第三步：反向滤波
    filter.reset();
    filter.process(numSamples, &temp);
    std::cout << "  - 反向滤波完成" << std::endl;
    
    // 第四步：再次反转得到最终结果
    std::reverse(temp, temp + numSamples);
    std::cout << "  - 最终反转完成" << std::endl;
    
    // 复制结果到输出
    memcpy(data, temp, numSamples * sizeof(float));
    delete[] temp;
    
    std::cout << "零相位滤波完成！" << std::endl;
}

// ===================== 信号保存函数 =====================

/**
 * @brief 保存信号到文件
 * @param data 信号数据
 * @param numSamples 样本数量
 * @param filepath 输出文件路径
 * @param precision 小数精度（默认6位）
 */
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

// ===================== ECG信号带通滤波处理函数 =====================

/**
 * @brief ECG信号带通滤波处理（Butterworth零相位滤波）
 * @param input_filepath 输入信号文件路径
 * @param output_filepath 输出信号文件路径
 * @param low_freq 低频截止频率 (Hz)
 * @param high_freq 高频截止频率 (Hz)
 * @param sample_rate 采样率 (Hz)
 * @param filter_order 滤波器阶数
 * @param original_weight 原始信号权重（默认0.2）
 * @param filtered_weight 滤波信号权重（默认0.8）
 * @param num_samples 样本数量（默认65000）
 */
void bandpass_filter(
    const std::string& input_filepath,
    const std::string& output_filepath,
    double low_freq,
    double high_freq,
    double sample_rate,
    int filter_order = 5,
    float original_weight = 0.2f,
    float filtered_weight = 0.8f,
    int num_samples = 65000
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ECG带通滤波器处理" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 显示参数
    std::cout << "\n【滤波器参数】" << std::endl;
    std::cout << "  阶数: " << filter_order << std::endl;
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
    
    // 创建巴特沃斯带通滤波器
    Dsp::SimpleFilter<Dsp::Butterworth::BandPass<5>, 1> filter;
    filter.setup(filter_order, sample_rate, center_frequency, bandwidth);
    
    std::cout << "\n【滤波器设置】" << std::endl;
    std::cout << "  类型: Butterworth 带通滤波器" << std::endl;
    std::cout << "  方法: 零相位滤波 (filtfilt)" << std::endl;
    
    // 分配内存
    float* original = new float[num_samples];
    float* filtered = new float[num_samples];
    float* combined = new float[num_samples];
    
    // 读取原始信号
    std::cout << "\n【读取信号】" << std::endl;
    std::ifstream input_file(input_filepath);
    if (!input_file.is_open()) {
        std::cerr << "错误：无法打开输入文件 " << input_filepath << std::endl;
        delete[] original;
        delete[] filtered;
        delete[] combined;
        return;
    }
    
    for (int i = 0; i < num_samples; ++i) {
        input_file >> original[i];
    }
    input_file.close();
    std::cout << "  输入文件: " << input_filepath << std::endl;
    std::cout << "  样本数: " << num_samples << std::endl;
    
    // 滤波处理
    std::cout << "\n【滤波处理】" << std::endl;
    memcpy(filtered, original, num_samples * sizeof(float));
    filtfilt(filter, filtered, num_samples);
    
    // 加权组合
    std::cout << "\n【加权组合】" << std::endl;
    for (int i = 0; i < num_samples; ++i) {
        combined[i] = original_weight * original[i] + filtered_weight * filtered[i];
    }
    std::cout << "  组合完成: " << original_weight << "*原始 + " 
              << filtered_weight << "*滤波" << std::endl;
    
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
    delete[] filtered;
    delete[] combined;
    
    std::cout << "\n处理完成！" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// ===================== 主函数 =====================

int main() {
    try {
        // ECG信号处理参数
        std::string input_file = "/home/yogsothoth/桌面/workspace-ppg/dataset/signal_100.txt";
        std::string output_file = "/home/yogsothoth/桌面/workspace-ppg/dataset/CPP_filtered_signal_20Hz.txt";
        
        // 调用ECG带通滤波处理
        bandpass_filter(
            input_file,           // 输入文件路径
            output_file,          // 输出文件路径
            5.0,                  // 低频截止 (Hz)
            20.0,                 // 高频截止 (Hz)
            360.0,                // 采样率 (Hz)
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
