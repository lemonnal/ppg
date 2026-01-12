#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <fstream>
#include "include/ecg_offline.hpp"

/*
 * ========================================================================
 * ECG QRS检测 - Pan-Tomkins算法 (离线版本)
 * ========================================================================
 * 
 * 本程序实现了基于Pan-Tomkins算法的QRS波检测器
 * 对应Python中的offline.py文件
 * 
 * 【主要功能】
 * 1. 带通滤波 (Butterworth滤波器，使用DSPFilters库)
 * 2. 微分 (5点中心差分)
 * 3. 平方
 * 4. 移动窗口积分
 * 5. 自适应阈值检测
 * 6. 心率计算 (RR间期、瞬时心率、平均心率等)
 * 
 * 【Pan-Tomkins算法流程】
 * ECG信号 → 带通滤波 → 微分 → 平方 → 移动窗口积分 → 阈值检测 → QRS波峰值
 * 
 * 【数据格式】
 * 输入: 文本文件，每行一个浮点数值
 * 输出: QRS波峰值位置、心率统计信息
 * ========================================================================
 */

int main() {
    try {
        // 固定参数
        std::string signal_file = "/home/yogsothoth/桌面/workspace-ppg/output_data/test_ecg_signal.txt";
        std::string signal_name = "MLII";
        float fs = 360.0f;
        size_t max_samples = 0;  // 0表示读取全部
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "ECG QRS检测 - Pan-Tomkins算法" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        std::cout << "\n【配置参数】" << std::endl;
        std::cout << "  导联名称: " << signal_name << std::endl;
        std::cout << "  采样率: " << fs << " Hz" << std::endl;
        std::cout << "  信号文件: " << signal_file << std::endl;
        if (max_samples > 0) {
            std::cout << "  最大样本数: " << max_samples << std::endl;
        }
        
        // ========== 步骤1: 读取ECG信号 ==========
        std::cout << "\n【步骤1: 读取ECG信号】" << std::endl;
        std::vector<float> signal;
        
        try {
            signal = ecg::read_ecg_signal(signal_file, max_samples);
            std::cout << "  ✓ 成功读取信号" << std::endl;
            std::cout << "  信号长度: " << signal.size() << " 样本" << std::endl;
            std::cout << "  时长: " << signal.size() / fs << " 秒" << std::endl;
            
            // 计算信号统计信息
            float min_val = *std::min_element(signal.begin(), signal.end());
            float max_val = *std::max_element(signal.begin(), signal.end());
            float mean_val = std::accumulate(signal.begin(), signal.end(), 0.0f) / signal.size();
            
            std::cout << "  信号范围: [" << min_val << ", " << max_val << "]" << std::endl;
            std::cout << "  平均值: " << mean_val << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "\n错误: " << e.what() << std::endl;
            std::cerr << "\n提示: 请确保信号文件存在且格式正确（每行一个数值）" << std::endl;
            return 1;
        }
        
        // ========== 步骤2: 创建QRS检测器 ==========
        std::cout << "\n【步骤2: 创建QRS检测器】" << std::endl;
        ecg::PanTomkinsQRSDetectorOffline detector(signal_name, fs);
        std::cout << "  ✓ 检测器已初始化" << std::endl;
        
        // ========== 步骤3: 执行QRS检测 ==========
        std::cout << "\n【步骤3: 执行QRS检测】" << std::endl;
        std::cout << "  正在处理..." << std::endl;
        
        std::vector<int> qrs_peaks = detector.detect_qrs_peaks(signal);
        
        std::cout << "  ✓ 检测完成" << std::endl;
        std::cout << "  检测到 " << qrs_peaks.size() << " 个R峰" << std::endl;
        
        if (qrs_peaks.empty()) {
            std::cout << "\n警告: 未检测到QRS波，请检查信号质量或调整参数" << std::endl;
            return 0;
        }
        
        // 显示前5个R峰位置
        std::cout << "\n  前5个R峰位置:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), qrs_peaks.size()); ++i) {
            std::cout << "    R" << (i + 1) << ": 样本 " << qrs_peaks[i] 
                     << " (时间: " << std::fixed << std::setprecision(3) 
                     << qrs_peaks[i] / fs << " s)" << std::endl;
        }
        
        // ========== 步骤4: 计算RR间期 ==========
        std::cout << "\n【步骤4: 计算RR间期】" << std::endl;
        std::vector<float> rr_intervals_sec;
        std::vector<int> rr_intervals_samples;
        detector.calculate_rr_intervals(qrs_peaks, rr_intervals_sec, rr_intervals_samples);
        
        if (!rr_intervals_sec.empty()) {
            std::cout << "  RR间期数量: " << rr_intervals_sec.size() << std::endl;
            
            float min_rr = *std::min_element(rr_intervals_sec.begin(), rr_intervals_sec.end());
            float max_rr = *std::max_element(rr_intervals_sec.begin(), rr_intervals_sec.end());
            float mean_rr = std::accumulate(rr_intervals_sec.begin(), rr_intervals_sec.end(), 0.0f) 
                          / rr_intervals_sec.size();
            
            std::cout << "  RR间期范围: " << std::fixed << std::setprecision(3) 
                     << min_rr << " - " << max_rr << " 秒" << std::endl;
            std::cout << "  平均RR间期: " << mean_rr << " 秒" << std::endl;
        }
        
        // ========== 步骤5: 计算瞬时心率 ==========
        std::cout << "\n【步骤5: 计算瞬时心率】" << std::endl;
        std::vector<float> inst_hr;
        std::vector<float> hr_times;
        detector.calculate_instantaneous_heart_rate(qrs_peaks, inst_hr, hr_times);
        
        if (!inst_hr.empty()) {
            std::cout << "  瞬时心率数量: " << inst_hr.size() << std::endl;
            
            float min_hr = *std::min_element(inst_hr.begin(), inst_hr.end());
            float max_hr = *std::max_element(inst_hr.begin(), inst_hr.end());
            
            std::cout << "  瞬时心率范围: " << std::fixed << std::setprecision(1) 
                     << min_hr << " - " << max_hr << " bpm" << std::endl;
            
            // 显示前5个瞬时心率值
            std::cout << "\n  前5个瞬时心率值:" << std::endl;
            for (size_t i = 0; i < std::min(size_t(5), inst_hr.size()); ++i) {
                std::cout << "    时间 " << std::fixed << std::setprecision(2) 
                         << hr_times[i] << " s: " << std::setprecision(1) 
                         << inst_hr[i] << " bpm" << std::endl;
            }
        }
        
        // ========== 步骤6: 计算平均心率 ==========
        std::cout << "\n【步骤6: 计算平均心率】" << std::endl;
        float avg_hr = 0.0f, std_hr = 0.0f;
        detector.calculate_average_heart_rate(qrs_peaks, avg_hr, std_hr);
        
        std::cout << "  平均心率: " << std::fixed << std::setprecision(1) 
                 << avg_hr << " ± " << std_hr << " bpm" << std::endl;
        
        // ========== 步骤7: 计算滑动窗口心率 ==========
        std::cout << "\n【步骤7: 计算滑动窗口心率】" << std::endl;
        int window_size = 10;
        std::vector<float> sliding_hr;
        std::vector<float> sliding_times;
        detector.calculate_sliding_window_heart_rate(window_size, qrs_peaks, 
                                                     sliding_hr, sliding_times);
        
        if (!sliding_hr.empty()) {
            std::cout << "  窗口大小: " << window_size << " 个心跳" << std::endl;
            std::cout << "  数据点数: " << sliding_hr.size() << std::endl;
            
            float min_sliding = *std::min_element(sliding_hr.begin(), sliding_hr.end());
            float max_sliding = *std::max_element(sliding_hr.begin(), sliding_hr.end());
            
            std::cout << "  心率范围: " << std::fixed << std::setprecision(1) 
                     << min_sliding << " - " << max_sliding << " bpm" << std::endl;
        }
        
        // ========== 步骤8: 获取完整统计信息 ==========
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "心率统计信息" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        ecg::HeartRateStats stats = detector.get_heart_rate_statistics(qrs_peaks);
        
        std::cout << "  总心跳数: " << stats.num_beats << std::endl;
        std::cout << "  平均心率: " << std::fixed << std::setprecision(1) 
                 << stats.mean_hr << " bpm" << std::endl;
        std::cout << "  标准差: " << stats.std_hr << " bpm" << std::endl;
        std::cout << "  最小心率: " << stats.min_hr << " bpm" << std::endl;
        std::cout << "  最大心率: " << stats.max_hr << " bpm" << std::endl;
        std::cout << "  中位数心率: " << stats.median_hr << " bpm" << std::endl;
        
        // ========== 步骤9: 保存处理结果（可选）==========
        std::cout << "\n【步骤9: 保存处理结果】" << std::endl;
        
        // 使用绝对路径保存
        std::string output_dir = "/home/yogsothoth/桌面/workspace-ppg/output_data/";
        
        std::string filtered_file = output_dir + "test_ecg_signal_filtered.txt";
        std::string integrated_file = output_dir + "test_ecg_signal_integrated.txt";
        std::string peaks_file = output_dir + "test_ecg_signal_peaks.txt";
        
        if (ecg::save_signal_to_file(detector.get_filtered_signal(), filtered_file)) {
            std::cout << "  ✓ 滤波信号已保存: " << filtered_file << std::endl;
        }
        
        if (ecg::save_signal_to_file(detector.get_integrated_signal(), integrated_file)) {
            std::cout << "  ✓ 积分信号已保存: " << integrated_file << std::endl;
        }
        
        // 保存峰值位置
        std::ofstream peaks_out(peaks_file);
        if (peaks_out.is_open()) {
            for (int peak : qrs_peaks) {
                peaks_out << peak << "\n";
            }
            peaks_out.close();
            std::cout << "  ✓ R峰位置已保存: " << peaks_file << std::endl;
        }
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "处理完成！" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

