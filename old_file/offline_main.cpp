#include <iostream>
#include <string>
#include <fstream>
#include "include/signal_io.hpp"
#include "include/ppg_filters.hpp"
#include "include/ppg_analysis.hpp"
#include "include/signal_utils.hpp"
#include "include/find_peaks.hpp"

/*
 * ========================================================================
 * PPG 光学类型与信号特征
 * ========================================================================
 * 
 * 【光学波长与后缀对应关系】
 * 后缀    光类型         波长范围           信号特征
 * ─────────────────────────────────────────────────────────────────────
 * _2     红光 (Red)      ~660nm          信号最强，标准差和振幅最大
 * _1     红外光 (IR)     ~880-940nm      信号中等
 * _3     绿光 (Green)    ~530nm          信号最弱，标准差和振幅最小
 * 
 * 【实验数据统计】
 * - 标准差排序: _2 (169211) > _1 (133211) > _3 (133)
 * - 振幅排序:   _2 (599750) > _1 (487750) > _3 (487)
 * 
 * 【光学原理】
 * 1. 红光 (660nm):   被血液中血红蛋白吸收最强 → 信号幅度最大
 * 2. 红外光 (880nm): 吸收中等 → 信号幅度中等
 * 3. 绿光 (530nm):   吸收最弱 → 信号幅度最小
 * 
 * 【SpO2 计算说明】
 * SpO2 (血氧饱和度) 测量需要使用红光和红外光的比值:
 * - 氧合血红蛋白 (HbO2) 对红光吸收较少，对红外光吸收较多
 * - 脱氧血红蛋白 (Hb) 对红光吸收较多，对红外光吸收较少
 * - 通过计算红光和红外光的AC/DC比值，可估算血氧饱和度
 * 
 * 本程序读取红光(_2)和红外光(_1)信号进行分析
 * ========================================================================
 */

int main() {
    try {
        // 读取文件名（支持多行格式）
        std::string record_name = "record.txt";
        std::ifstream record_file("/home/yogsothoth/桌面/workspace-ppg/" + record_name);
        std::vector<std::string> file_list;

        if (!record_file.is_open()) {
            std::cerr << "错误：无法打开 " << record_name << " 文件" << std::endl;
            return 1;
        }

        // 逐行读取文件名
        std::string line;
        while (std::getline(record_file, line)) {
            // 去除空白字符
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            // 跳过空行
            if (!line.empty()) {
                file_list.push_back(line);
            }
        }
        record_file.close();

        if (file_list.empty()) {
            std::cerr << "错误：" << record_name << " 文件为空" << std::endl;
            return 1;
        }

        std::cout << "📂 从 " << record_name << " 读取到 " << file_list.size() << " 个文件:" << std::endl;
        for (size_t i = 0; i < file_list.size(); ++i) {
            std::cout << "   " << (i + 1) << ". " << file_list[i] << std::endl;
        }
        std::cout << std::string(60, '=') << std::endl;

        for (const std::string& file_name : file_list) {
            // 构建红光和红外光文件路径
            std::string red_file = "/home/yogsothoth/桌面/workspace-ppg/DataSet/PPG-BP/" + file_name + "_2.txt";    // 红光 (660nm)
            std::string ir_file = "/home/yogsothoth/桌面/workspace-ppg/DataSet/PPG-BP/" + file_name + "_1.txt";     // 红外光 (880nm)
            
            std::string red_filtered_oneway = "/home/yogsothoth/桌面/workspace-ppg/output_data/" + file_name + "_2_filtered_oneway.txt";
            std::string ir_filtered_oneway = "/home/yogsothoth/桌面/workspace-ppg/output_data/" + file_name + "_1_filtered_oneway.txt";

            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "PPG信号处理 - red/ir 双通道分析" << std::endl;
            std::cout << "文件: " << file_name << std::endl;
            std::cout << std::string(60, '=') << std::endl;
            std::cout << "\n【滤波器参数配置】" << std::endl;
            std::cout << "  - 类型: Butterworth 带通滤波器" << std::endl;
            std::cout << "  - 阶数: 3" << std::endl;
            std::cout << "  - 低频截止: 0.5 Hz (去除基线漂移)" << std::endl;
            std::cout << "  - 高频截止: 20 Hz (去除高频噪声)" << std::endl;
            std::cout << "  - 通带范围: 0.5-20 Hz (保留心率相关频率)" << std::endl;
            std::cout << "  - 采样率: 1000 Hz (PPG-BP数据集)" << std::endl;
            std::cout << std::string(60, '=') << "\n" << std::endl;

            // ========== 步骤1: 读取红光信号 ==========
            std::cout << "【步骤1: 读取红光信号 (660nm)】" << std::endl;
            std::vector<float> red_signal = ppg::read_signal_from_file(red_file, 2100);
            
            if (red_signal.empty()) {
                std::cerr << "错误：无法读取红光信号 " << red_file << std::endl;
                continue;  // 跳过当前文件，继续处理下一个
            }
            
            std::cout << "  信号长度: " << red_signal.size() << " 样本" << std::endl;
            std::cout << "  信号范围: [" << *std::min_element(red_signal.begin(), red_signal.end())
                    << ", " << *std::max_element(red_signal.begin(), red_signal.end()) << "]" << std::endl;
            
            // ========== 步骤1b: 读取红外光信号 ==========
            std::cout << "\n【步骤1b: 读取红外光信号 (880nm)】" << std::endl;
            std::vector<float> ir_signal = ppg::read_signal_from_file(ir_file, 2100);
            
            if (ir_signal.empty()) {
                std::cerr << "错误：无法读取红外光信号 " << ir_file << std::endl;
                continue;  // 跳过当前文件，继续处理下一个
            }
            
            std::cout << "  信号长度: " << ir_signal.size() << " 样本" << std::endl;
            std::cout << "  信号范围: [" << *std::min_element(ir_signal.begin(), ir_signal.end())
                    << ", " << *std::max_element(ir_signal.begin(), ir_signal.end()) << "]" << std::endl;
            
            // ========== 步骤2: 单向IIR滤波 ==========
            
            std::vector<float> red_filtered, ir_filtered;
            std::string red_output_file, ir_output_file;
            

            std::cout << "\n【步骤2: 单向IIR滤波 - 红光】" << std::endl;
            std::cout << "  优点: 低延迟，逐样本处理，适合实时" << std::endl;
            std::cout << "  缺点: 有相位失真（群延迟）\n" << std::endl;
            
            // 应用单向IIR滤波 - 红光
            red_filtered = ppg::apply_bandpass_oneway(
                red_signal,      // 输入信号数组
                0.5,             // 低频截止 (Hz)
                20.0,            // 高频截止 (Hz)
                1000.0,          // 采样率 (Hz)
                3,               // 滤波器阶数
                true             // 使用均值初始化预热（减少瞬态响应）
            );
            
            // 应用单向IIR滤波 - 红外光
            std::cout << "【步骤2b: 单向IIR滤波 - 红外光】" << std::endl;
            ir_filtered = ppg::apply_bandpass_oneway(
                ir_signal,       // 输入信号数组
                0.5,             // 低频截止 (Hz)
                20.0,            // 高频截止 (Hz)
                1000.0,          // 采样率 (Hz)
                3,               // 滤波器阶数
                true             // 使用均值初始化预热（减少瞬态响应）
            );
            
            red_output_file = red_filtered_oneway;
            ir_output_file = ir_filtered_oneway;
            
            // 保存滤波结果
            ppg::save_signal_to_file(red_filtered, red_output_file);
            std::cout << "  ✓ 红光滤波结果已保存" << std::endl;
            ppg::save_signal_to_file(ir_filtered, ir_output_file);
            std::cout << "  ✓ 红外光滤波结果已保存" << std::endl;

            // ========== 步骤3: 峰值检测 - 红光 ==========
            std::cout << "\n【步骤3: 峰值检测 - 红光】" << std::endl;
            std::vector<int> red_peaks, red_valleys;
            float red_ac_component = 0.0f;
            ppg::detect_peaks_and_valleys(red_filtered, 1000.0, 0.4, red_peaks, red_valleys, red_ac_component);

            // ========== 步骤3b: 峰值检测 - 红外光 ==========
            std::cout << "【步骤3b: 峰值检测 - 红外光】" << std::endl;
            std::vector<int> ir_peaks, ir_valleys;
            float ir_ac_component = 0.0f;
            ppg::detect_peaks_and_valleys(ir_filtered, 1000.0, 0.4, ir_peaks, ir_valleys, ir_ac_component);

            // ========== 步骤4: 心率计算 - 红光 ==========
            std::cout << "\n【步骤4: 心率计算 - 红光】" << std::endl;
            float red_heart_rate = 0.0f;
            float red_hrv = 0.0f;
            ppg::calculate_heart_rate(red_peaks, 1000.0, red_heart_rate, red_hrv);

            // ========== 步骤4b: 心率计算 - 红外光 ==========
            std::cout << "【步骤4b: 心率计算 - 红外光】" << std::endl;
            float ir_heart_rate = 0.0f;
            float ir_hrv = 0.0f;
            ppg::calculate_heart_rate(ir_peaks, 1000.0, ir_heart_rate, ir_hrv);

            // ========== 步骤5: SpO2估算 (双通道：红光+红外光) ==========
            float spo2 = 0.0f;
            float ratio = 0.0f;
            ppg::calculate_spo2_dual_channel(
                red_signal, red_filtered, red_ac_component,
                ir_signal, ir_filtered, ir_ac_component,
                spo2, ratio
            );

            // ========== 步骤6: 信号统计 ==========
            std::cout << "\n【红光信号统计】" << std::endl;
            ppg::print_signal_statistics(red_signal, red_filtered);
            std::cout << "\n【红外光信号统计】" << std::endl;
            ppg::print_signal_statistics(ir_signal, ir_filtered);
        }
    } catch (const std::exception& e) {
        std::cerr << "\n错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

