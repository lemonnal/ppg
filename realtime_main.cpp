#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <cmath>
#include "include/realtime_filter.hpp"
#include "include/ppg_analysis.hpp"

/**
 * @brief 实时PPG信号处理系统 - 双通道版本
 *
 * 模拟嵌入式设备实时系统环境：
 * - 从文件逐样本读取双通道数据（红光+红外光）
 * - 使用单向IIR滤波器实时处理
 * - 维护滑动窗口进行分析
 * - 定期计算心率和SpO2
 * - 使用int16缓冲区优化内存使用
 */

int main()
{
    try
    {
        // ==================== 系统配置 ====================
        const std::string file_name = "259";
        const std::string red_file = "/home/yogsothoth/桌面/workspace-ppg/aaaPyTest/concat_" + file_name + "_3.txt";    // 红光 (660nm)2
        const std::string ir_file = "/home/yogsothoth/桌面/workspace-ppg/aaaPyTest/concat_" + file_name + "_3.txt";     // 红外光 (880nm)1
        
        const double SAMPLE_RATE = 1000.0; // 采样率 1000 Hz
        const double LOW_FREQ = 0.5;      // 低频截止
        const double HIGH_FREQ = 20.0;    // 高频截止
        const int FILTER_ORDER = 3;       // 滤波器阶数

        // 缓冲区配置（模拟嵌入式系统的内存限制）
        const size_t ANALYSIS_WINDOW = 2100;                // 分析窗口：2.1秒
        const size_t BUFFER_SIZE = ANALYSIS_WINDOW + 200;   // 2.3秒的数据
        const size_t UPDATE_INTERVAL = ANALYSIS_WINDOW / 2; // 每1.05秒更新一次分析

        // 是否实时模拟（添加延迟）
        const bool SIMULATE_REALTIME = true;   // true: 按实际采样率添加延迟
        const double SAMPLE_INTERVAL_MS = 1.0; // 1ms per sample @ 1000Hz

        std::cout << "\n"
                  << std::string(70, '=') << std::endl;
        std::cout << "    实时PPG信号处理系统 - 双通道嵌入式模拟模式" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::cout << "\n【系统配置】" << std::endl;
        std::cout << "  数据文件: " << file_name << std::endl;
        std::cout << "  红光数据: " << red_file << std::endl;
        std::cout << "  红外光数据: " << ir_file << std::endl;
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
        std::cout << "  内存模式: 16位整型 (节省内存)" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        // ==================== 初始化组件 ====================
        std::cout << "\n【初始化系统组件】" << std::endl;

        // 1. 创建双通道实时滤波器
        ppg::RealtimeFilter filter_red(LOW_FREQ, HIGH_FREQ, SAMPLE_RATE, FILTER_ORDER);
        ppg::RealtimeFilter filter_ir(LOW_FREQ, HIGH_FREQ, SAMPLE_RATE, FILTER_ORDER);
        std::cout << "  ✓ 双通道滤波器创建完成 (红光 + 红外光)" << std::endl;

        // 2. 创建双通道数据缓冲区 (int16)
        ppg::RealtimeBufferInt16 raw_buffer_red(BUFFER_SIZE);      // 红光原始信号
        ppg::RealtimeBufferInt16 raw_buffer_ir(BUFFER_SIZE);       // 红外光原始信号
        ppg::RealtimeBufferInt16 filtered_buffer_red(BUFFER_SIZE); // 红光滤波信号
        ppg::RealtimeBufferInt16 filtered_buffer_ir(BUFFER_SIZE);  // 红外光滤波信号

        std::cout << "  ✓ 双通道数据缓冲区创建完成 (16位整型: "
                  << (BUFFER_SIZE * 4 * 2) / 1024.0 << "KB)" << std::endl;

        // 3. 打开双通道数据文件
        std::ifstream red_stream(red_file);
        std::ifstream ir_stream(ir_file);
        if (!red_stream.is_open())
        {
            std::cerr << "\n错误：无法打开红光数据文件 " << red_file << std::endl;
            return 1;
        }
        if (!ir_stream.is_open())
        {
            std::cerr << "\n错误：无法打开红外光数据文件 " << ir_file << std::endl;
            return 1;
        }
        std::cout << "  ✓ 双通道数据文件打开成功" << std::endl;

        // 4. 预读取一些样本用于滤波器预热
        std::cout << "\n【滤波器预热】" << std::endl;
        std::vector<float> warmup_samples_red, warmup_samples_ir;
        std::string line_red, line_ir;
        size_t warmup_count = 100;
        float warmup_sum_red = 0.0f, warmup_sum_ir = 0.0f;

        while (warmup_samples_red.size() < warmup_count && 
               std::getline(red_stream, line_red) && 
               std::getline(ir_stream, line_ir))
        {
            try
            {
                float red_value = std::stof(line_red);
                float ir_value = std::stof(line_ir);
                
                warmup_samples_red.push_back(red_value);
                warmup_samples_ir.push_back(ir_value);
                warmup_sum_red += red_value;
                warmup_sum_ir += ir_value;
            }
            catch (...)
            {
                continue;
            }
        }

        if (warmup_samples_red.empty())
        {
            std::cerr << "错误：无法读取数据" << std::endl;
            return 1;
        }

        float initial_mean_red = warmup_sum_red / warmup_samples_red.size();
        float initial_mean_ir = warmup_sum_ir / warmup_samples_ir.size();
        filter_red.warmup(initial_mean_red, 100);
        filter_ir.warmup(initial_mean_ir, 100);
        std::cout << "  ✓ 双通道滤波器预热完成 (红光均值: " << initial_mean_red 
                  << ", 红外光均值: " << initial_mean_ir << ")" << std::endl;

        // 将预热样本重新放回文件流（通过重新定位）
        red_stream.clear();
        red_stream.seekg(0, std::ios::beg);
        ir_stream.clear();
        ir_stream.seekg(0, std::ios::beg);

        // ==================== 实时处理主循环 ====================
        std::cout << "\n"
                  << std::string(70, '=') << std::endl;
        std::cout << "开始实时数据处理..." << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        size_t sample_count = 0;
        size_t last_analysis_count = 0;
        int analysis_count = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        // 逐样本读取并处理双通道数据
        while (std::getline(red_stream, line_red) && std::getline(ir_stream, line_ir))
        {
            int16_t raw_sample_red, raw_sample_ir;
            try
            {
                raw_sample_red = static_cast<int16_t>(std::stoi(line_red));
                raw_sample_ir = static_cast<int16_t>(std::stoi(line_ir));
            }
            catch (...)
            {
                continue; // 跳过无效数据
            }

            // 步骤1: 双通道实时滤波
            float raw_red_float = static_cast<float>(raw_sample_red);
            float raw_ir_float = static_cast<float>(raw_sample_ir);
            
            float filtered_red_float = filter_red.process_sample(raw_red_float);
            float filtered_ir_float = filter_ir.process_sample(raw_ir_float);

            // 四舍五入转换为整型（损失小数精度但节省内存）
            int16_t filtered_red_int = static_cast<int16_t>(std::round(filtered_red_float));
            int16_t filtered_ir_int = static_cast<int16_t>(std::round(filtered_ir_float));

            // 步骤2: 添加到双通道缓冲区
            raw_buffer_red.push(raw_sample_red);
            raw_buffer_ir.push(raw_sample_ir);
            filtered_buffer_red.push(filtered_red_int);
            filtered_buffer_ir.push(filtered_ir_int);

            sample_count++;

            // 步骤3: 定期进行信号分析
            if (sample_count >= ANALYSIS_WINDOW &&
                (sample_count - last_analysis_count) >= UPDATE_INTERVAL)
            {

                analysis_count++;
                last_analysis_count = sample_count;

                // 获取双通道分析窗口数据
                size_t start_idx = 0;
                if (filtered_buffer_red.size() > ANALYSIS_WINDOW)
                {
                    start_idx = filtered_buffer_red.size() - ANALYSIS_WINDOW;
                }

                // 获取滤波数据和原始数据
                std::vector<float> filtered_data_red = filtered_buffer_red.get_data_float(
                    start_idx, ANALYSIS_WINDOW);
                std::vector<float> raw_data_red = raw_buffer_red.get_data_float(
                    start_idx, ANALYSIS_WINDOW);
                std::vector<float> filtered_data_ir = filtered_buffer_ir.get_data_float(
                    start_idx, ANALYSIS_WINDOW);
                std::vector<float> raw_data_ir = raw_buffer_ir.get_data_float(
                    start_idx, ANALYSIS_WINDOW);

                // 峰值检测和AC分量计算 - 红光通道
                std::vector<int> red_peaks, red_valleys;
                float red_ac_component = 0.0f;
                ppg::detect_peaks_and_valleys(
                    filtered_data_red,
                    SAMPLE_RATE,
                    0.4, // 最小峰值间隔0.4秒
                    red_peaks,
                    red_valleys,
                    red_ac_component);

                // 峰值检测和AC分量计算 - 红外光通道
                std::vector<int> ir_peaks, ir_valleys;
                float ir_ac_component = 0.0f;
                ppg::detect_peaks_and_valleys(
                    filtered_data_ir,
                    SAMPLE_RATE,
                    0.4,
                    ir_peaks,
                    ir_valleys,
                    ir_ac_component);

                // 心率计算（使用红光通道的峰值）
                float heart_rate = 0.0f;
                float hrv = 0.0f;
                bool hr_valid = ppg::calculate_heart_rate(
                    red_peaks,
                    SAMPLE_RATE,
                    heart_rate,
                    hrv);

                // SpO2计算（使用双通道数据）
                float spo2 = 0.0f;
                float ratio = 0.0f;
                bool spo2_valid = ppg::calculate_spo2_dual_channel(
                    raw_data_red,
                    filtered_data_red,
                    red_ac_component,
                    raw_data_ir,
                    filtered_data_ir,
                    ir_ac_component,
                    spo2,
                    ratio);

                // 输出结果
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   current_time - start_time)
                                   .count();

                std::cout << "\n[分析 #" << analysis_count << "] ";
                std::cout << "样本: " << sample_count << " | ";
                std::cout << "时间: " << elapsed / 1000.0 << "s | ";
                std::cout << "缓冲区: " << filtered_buffer_red.size() << "/" << BUFFER_SIZE << std::endl;

                std::cout << "  峰值数(红光): " << red_peaks.size() << " (红外光): " << ir_peaks.size() << " | ";
                std::cout << "谷值数(红光): " << red_valleys.size() << " (红外光): " << ir_valleys.size() << std::endl;
                std::cout << "  AC(红光): " << red_ac_component << " | AC(红外光): " << ir_ac_component << std::endl;

                if (hr_valid)
                {
                    std::cout << "  ❤️  心率: " << heart_rate << " BPM | ";
                    std::cout << "HRV: " << hrv << " ms" << std::endl;
                }
                else
                {
                    std::cout << "  ❤️  心率: 无效 (峰值不足)" << std::endl;
                }

                if (spo2_valid)
                {
                    std::cout << "  🫁 SpO2: " << spo2 << " % | ";
                    std::cout << "R: " << ratio << std::endl;
                }
                else
                {
                    std::cout << "  🫁 SpO2: 无效 (信号质量不足)" << std::endl;
                }

                std::cout << std::string(70, '-') << std::endl;
            }

            // 模拟实时延迟（可选）
            if (SIMULATE_REALTIME && sample_count % 10 == 0)
            {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(static_cast<int>(SAMPLE_INTERVAL_MS * 1000 * 10)));
            }

            // 定期显示进度（每5000个样本）
            if (sample_count % 5000 == 0)
            {
                std::cout << "处理进度: " << sample_count << " 样本..." << std::endl;
            }
        }

        // ==================== 处理完成 ====================
        red_stream.close();
        ir_stream.close();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  end_time - start_time)
                                  .count();

        std::cout << "\n"
                  << std::string(70, '=') << std::endl;
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
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n❌ 错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
