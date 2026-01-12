#include "signal_io.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

namespace ppg {

std::vector<float> read_signal_from_file(
    const std::string& filepath, 
    int max_samples
) {
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

void save_signal_to_file(
    const std::vector<float>& signal, 
    const std::string& filepath, 
    int precision
) {
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

} // namespace ppg

