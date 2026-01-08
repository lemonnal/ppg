#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <deque>
#include <chrono>
#include <thread>
#include <cmath>
#include "include/realtime_filter.hpp"
#include "include/ppg_analysis.hpp"
#include "include/find_peaks.hpp"

// ==================== 内存优化配置 ====================
// 使用整型缓冲区可节省 50% 内存（24KB → 12KB）
// 代价：滤波后的数据会损失小数精度
#define USE_INT16_BUFFER 1  // 1=使用整型(节省内存), 0=使用浮点(保持精度)

/**
 * @brief 实时PPG信号处理系统
 * 
 * 模拟嵌入式设备实时系统环境：
 * - 从文件逐样本读取数据
 * - 使用单向IIR滤波器实时处理
 * - 维护滑动窗口进行分析
 * - 定期计算心率和SpO2
 */

int main() {
    try {
        // ==================== 系统配置 ====================
        const std::string data_file = "/home/yogsothoth/桌面/workspace-ppg/aaaPyTest/concat_259_3.txt";
        const double SAMPLE_RATE = 500.0;  // 采样率 1000 Hz
        const double LOW_FREQ = 0.5;        // 低频截止
        const double HIGH_FREQ = 20.0;      // 高频截止
        const int FILTER_ORDER = 3;         // 滤波器阶数
        
        // 缓冲区配置（模拟嵌入式系统的内存限制）
        const size_t ANALYSIS_WINDOW = 2100; // 分析窗口：2.1秒（与原始代码一致）
        const size_t BUFFER_SIZE = ANALYSIS_WINDOW + 200;     // 2.3秒的数据 (2300样本 @ 1000Hz)
        const size_t UPDATE_INTERVAL = ANALYSIS_WINDOW / 2;  // 每1.2秒更新一次分析
        
        // 是否实时模拟（添加延迟）
        const bool SIMULATE_REALTIME = true;  // true: 按实际采样率添加延迟
        const double SAMPLE_INTERVAL_MS = 1.0; // 1ms per sample @ 1000Hz
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "    实时PPG信号处理系统 - 嵌入式模拟模式" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        std::cout << "\n【系统配置】" << std::endl;
        std::cout << "  数据源: " << data_file << std::endl;
        std::cout << "  采样率: " << SAMPLE_RATE << " Hz" << std::endl;
        std::cout << "  滤波器: Butterworth 带通 (" << LOW_FREQ << "-" << HIGH_FREQ << " Hz)" << std::endl;
        std::cout << "  滤波器阶数: " << FILTER_ORDER << std::endl;
        std::cout << "  数据缓冲区: " << BUFFER_SIZE << " 样本 (" 
                  << BUFFER_SIZE / SAMPLE_RATE << " 秒)" << std::endl;
        std::cout << "  分析窗口: " << ANALYSIS_WINDOW << " 样本 (" 
                  << ANALYSIS_WINDOW / SAMPLE_RATE << " 秒)" << std::endl;
        std::cout << "  更新间隔: " << UPDATE_INTERVAL << " 样本 (" 
                  << UPDATE_INTERVAL / SAMPLE_RATE << " 秒)" << std::endl;
        std::cout << "  实时模拟: " << (SIMULATE_REALTIME ? "启用" : "禁用") << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        // ==================== 初始化组件 ====================
        std::cout << "\n【初始化系统组件】" << std::endl;
        
        // 1. 创建实时滤波器
        ppg::RealtimeFilter filter(LOW_FREQ, HIGH_FREQ, SAMPLE_RATE, FILTER_ORDER);
        
        // 2. 创建数据缓冲区
#if USE_INT16_BUFFER
        ppg::RealtimeBufferInt16 raw_buffer(BUFFER_SIZE);      // 原始信号缓冲区 (int16)
        ppg::RealtimeBufferInt16 filtered_buffer(BUFFER_SIZE); // 滤波信号缓冲区 (int16)
        
        std::cout << "  ✓ 数据缓冲区创建完成 (16位整型: " 
                  << (BUFFER_SIZE * 2 * 2) / 1024.0 << "KB)" << std::endl;
#else
        ppg::RealtimeBuffer raw_buffer(BUFFER_SIZE);      // 原始信号缓冲区 (float)
        ppg::RealtimeBuffer filtered_buffer(BUFFER_SIZE); // 滤波信号缓冲区 (float)
        
        std::cout << "  ✓ 数据缓冲区创建完成 (32位浮点: " 
                  << (BUFFER_SIZE * 2 * 4) / 1024.0 << " KB)" << std::endl;
#endif
        
        // 3. 打开数据文件
        std::ifstream data_stream(data_file);
        if (!data_stream.is_open()) {
            std::cerr << "\n错误：无法打开数据文件 " << data_file << std::endl;
            return 1;
        }
        std::cout << "  ✓ 数据文件打开成功" << std::endl;
        
        // 4. 预读取一些样本用于滤波器预热
        std::cout << "\n【滤波器预热】" << std::endl;
        std::vector<float> warmup_samples;
        std::string line;
        size_t warmup_count = 100;
        float warmup_sum = 0.0f;
        
        while (warmup_samples.size() < warmup_count && std::getline(data_stream, line)) {
            try {
                float value = std::stof(line);
                warmup_samples.push_back(value);
                warmup_sum += value;
            } catch (...) {
                continue;
            }
        }
        
        if (warmup_samples.empty()) {
            std::cerr << "错误：无法读取数据" << std::endl;
            return 1;
        }
        
        float initial_mean = warmup_sum / warmup_samples.size();
        filter.warmup(initial_mean, 100);
        
        // 将预热样本重新放回文件流（通过重新定位）
        data_stream.clear();
        data_stream.seekg(0, std::ios::beg);
        
        // ==================== 实时处理主循环 ====================
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "开始实时数据处理..." << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        size_t sample_count = 0;
        size_t last_analysis_count = 0;
        int analysis_count = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 逐样本读取并处理
        while (std::getline(data_stream, line)) {
#if USE_INT16_BUFFER
            // 整型缓冲区模式：节省内存但损失精度
            int16_t raw_sample;
            try {
                raw_sample = static_cast<int16_t>(std::stoi(line));
            } catch (...) {
                continue;  // 跳过无效数据
            }
            
            // 步骤1: 实时滤波（需要转换为float）
            float raw_sample_float = static_cast<float>(raw_sample);
            float filtered_sample_float = filter.process_sample(raw_sample_float);
            
            // 四舍五入转换为整型（损失小数精度但节省内存）
            int16_t filtered_sample_int = static_cast<int16_t>(
                std::round(filtered_sample_float)
            );
            
            // 步骤2: 添加到缓冲区
            raw_buffer.push(raw_sample);
            filtered_buffer.push(filtered_sample_int);
#else
            // 浮点缓冲区模式：保持精度
            float raw_sample;
            try {
                raw_sample = std::stof(line);
            } catch (...) {
                continue;  // 跳过无效数据
            }
            
            // 步骤1: 实时滤波（单向IIR）
            float filtered_sample = filter.process_sample(raw_sample);
            
            // 步骤2: 添加到缓冲区
            raw_buffer.push(raw_sample);
            filtered_buffer.push(filtered_sample);
#endif
            
            sample_count++;
            
            // 步骤3: 定期进行信号分析
            if (sample_count >= ANALYSIS_WINDOW && 
                (sample_count - last_analysis_count) >= UPDATE_INTERVAL) {
                
                analysis_count++;
                last_analysis_count = sample_count;
                
#if USE_INT16_BUFFER
                // 整型缓冲区：只获取需要的窗口数据并转换为浮点
                size_t start_idx = 0;
                if (filtered_buffer.size() > ANALYSIS_WINDOW) {
                    start_idx = filtered_buffer.size() - ANALYSIS_WINDOW;
                }
                
                // 直接获取指定范围的浮点数据，减少内存拷贝
                std::vector<float> filtered_data = filtered_buffer.get_data_float(
                    start_idx, ANALYSIS_WINDOW
                );
                std::vector<float> raw_data = raw_buffer.get_data_float(
                    start_idx, ANALYSIS_WINDOW
                );
#else
                // 浮点缓冲区：获取当前窗口数据
                std::vector<float> raw_data = raw_buffer.get_data();
                std::vector<float> filtered_data = filtered_buffer.get_data();
                
                // 只使用最近的ANALYSIS_WINDOW个样本
                if (filtered_data.size() > ANALYSIS_WINDOW) {
                    size_t start_idx = filtered_data.size() - ANALYSIS_WINDOW;
                    filtered_data = std::vector<float>(
                        filtered_data.begin() + start_idx, 
                        filtered_data.end()
                    );
                    raw_data = std::vector<float>(
                        raw_data.begin() + start_idx, 
                        raw_data.end()
                    );
                }
#endif
                
                // 峰值检测
                std::vector<int> peaks, valleys;
                float ac_component = 0.0f;
                ppg::detect_peaks_and_valleys(
                    filtered_data, 
                    SAMPLE_RATE, 
                    0.4,  // 最小峰值间隔0.4秒
                    peaks, 
                    valleys, 
                    ac_component
                );
                
                // 心率计算
                float heart_rate = 0.0f;
                float hrv = 0.0f;
                bool hr_valid = ppg::calculate_heart_rate(
                    peaks, 
                    SAMPLE_RATE, 
                    heart_rate, 
                    hrv
                );
                
                // SpO2计算
                float spo2 = 0.0f;
                float ratio = 0.0f;
                bool spo2_valid = ppg::calculate_spo2_from_ppg(
                    raw_data,
                    filtered_data,
                    peaks,
                    valleys,
                    ac_component,
                    spo2,
                    ratio
                );
                
                // 输出结果
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - start_time
                ).count();
                
                std::cout << "\n[分析 #" << analysis_count << "] ";
                std::cout << "样本: " << sample_count << " | ";
                std::cout << "时间: " << elapsed / 1000.0 << "s | ";
                std::cout << "缓冲区: " << filtered_buffer.size() << "/" << BUFFER_SIZE << std::endl;
                
                std::cout << "  峰值数: " << peaks.size() << " | ";
                std::cout << "谷值数: " << valleys.size() << " | ";
                std::cout << "AC: " << ac_component << std::endl;
                
                if (hr_valid) {
                    std::cout << "  ❤️  心率: " << heart_rate << " BPM | ";
                    std::cout << "HRV: " << hrv << " ms" << std::endl;
                } else {
                    std::cout << "  ❤️  心率: 无效 (峰值不足)" << std::endl;
                }
                
                if (spo2_valid) {
                    std::cout << "  🫁 SpO2: " << spo2 << " % | ";
                    std::cout << "R: " << ratio << std::endl;
                } else {
                    std::cout << "  🫁 SpO2: 无效 (信号质量不足)" << std::endl;
                }
                
                std::cout << std::string(70, '-') << std::endl;
            }
            
            // 模拟实时延迟（可选）
            if (SIMULATE_REALTIME && sample_count % 10 == 0) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(static_cast<int>(SAMPLE_INTERVAL_MS * 1000 * 10))
                );
            }
            
            // 定期显示进度（每5000个样本）
            if (sample_count % 5000 == 0) {
                std::cout << "处理进度: " << sample_count << " 样本..." << std::endl;
            }
        }
        
        // ==================== 处理完成 ====================
        data_stream.close();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time
        ).count();
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "实时处理完成！" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "\n【处理统计】" << std::endl;
        std::cout << "  总样本数: " << sample_count << std::endl;
        std::cout << "  总时长: " << sample_count / SAMPLE_RATE << " 秒" << std::endl;
        std::cout << "  处理耗时: " << total_duration / 1000.0 << " 秒" << std::endl;
        std::cout << "  处理速度: " << (sample_count / (total_duration / 1000.0)) << " 样本/秒" << std::endl;
        std::cout << "  实时因子: " << (sample_count / SAMPLE_RATE) / (total_duration / 1000.0) << "x" << std::endl;
        std::cout << "  分析次数: " << analysis_count << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ 错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

