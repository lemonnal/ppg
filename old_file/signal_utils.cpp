#include "signal_utils.hpp"
#include <iostream>
#include <iomanip>

namespace ppg {

void print_signal_statistics(
    const std::vector<float>& input_signal,
    const std::vector<float>& filtered_signal
) {
    std::cout << "\n【信号统计】" << std::endl;
    
    float input_mean = 0, filtered_mean = 0;
    float input_energy = 0, filtered_energy = 0;
    
    for (float val : input_signal) {
        input_mean += val;
        input_energy += val * val;
    }
    input_mean /= input_signal.size();
    input_energy /= input_signal.size();
    
    for (float val : filtered_signal) {
        filtered_mean += val;
        filtered_energy += val * val;
    }
    filtered_mean /= filtered_signal.size();
    filtered_energy /= filtered_signal.size();
    
    std::cout << "  原始信号 - 均值: " << std::fixed << std::setprecision(2) 
              << input_mean << ", 能量: " << input_energy << std::endl;
    std::cout << "  滤波信号 - 均值: " << filtered_mean 
              << ", 能量: " << filtered_energy << std::endl;
}

} // namespace ppg

