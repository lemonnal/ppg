#include <iostream>
#include <string>
#include "include/signal_io.hpp"
#include "include/ppg_filters.hpp"
#include "include/ppg_analysis.hpp"
#include "include/signal_utils.hpp"
#include "include/find_peaks.hpp"

int main() {
    try {
        // PPG信号处理参数
        std::string input_file = "/home/yogsothoth/桌面/workspace-ppg/DataSet/PPG-BP/2_1.txt";
        std::string filtered_file_zerophase = "/home/yogsothoth/桌面/workspace-ppg/DSPFilter/PPG-BP/2_1_filtered_zerophase.txt";
        std::string filtered_file_oneway = "/home/yogsothoth/桌面/workspace-ppg/DSPFilter/PPG-BP/2_1_filtered_oneway.txt";

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "PPG信号处理 - 带通滤波" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "\n【滤波器参数配置】" << std::endl;
        std::cout << "  - 类型: Butterworth 带通滤波器" << std::endl;
        std::cout << "  - 阶数: 3" << std::endl;
        std::cout << "  - 低频截止: 0.5 Hz (去除基线漂移)" << std::endl;
        std::cout << "  - 高频截止: 20 Hz (去除高频噪声)" << std::endl;
        std::cout << "  - 通带范围: 0.5-20 Hz (保留心率相关频率)" << std::endl;
        std::cout << "  - 采样率: 1000 Hz (PPG-BP数据集)" << std::endl;
        std::cout << std::string(60, '=') << "\n" << std::endl;

        // ========== 步骤1: 读取信号 ==========
        std::cout << "【步骤1: 读取信号】" << std::endl;
        std::vector<float> input_signal = ppg::read_signal_from_file(input_file, 2100);
        
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
            filtered_signal = ppg::apply_bandpass_zerophase(
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
            filtered_signal = ppg::apply_bandpass_oneway(
                input_signal,    // 输入信号数组
                0.5,             // 低频截止 (Hz)
                20.0,            // 高频截止 (Hz)
                1000.0,          // 采样率 (Hz)
                3,               // 滤波器阶数
                true             // 使用均值初始化预热（减少瞬态响应）
            );
            
            output_file = filtered_file_oneway;
        }
        
        // 保存滤波结果
        ppg::save_signal_to_file(filtered_signal, output_file);
        std::cout << "  ✓ 滤波结果已保存" << std::endl;

        // ========== 步骤3: 峰值检测 ==========
        std::vector<int> peaks, valleys;
        float ac_component = 0.0f;
        ppg::detect_peaks_and_valleys(filtered_signal, 1000.0, 0.4, peaks, valleys, ac_component);

        // ========== 步骤4: SpO2估算 ==========
        float spo2 = 0.0f;
        float ratio = 0.0f;
        ppg::calculate_spo2_from_ppg(input_signal, filtered_signal, peaks, valleys,
                                ac_component, spo2, ratio);

        // ========== 步骤5: 信号统计 ==========
        ppg::print_signal_statistics(input_signal, filtered_signal);

    } catch (const std::exception& e) {
        std::cerr << "\n错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

