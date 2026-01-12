#ifndef SIGNAL_IO_HPP
#define SIGNAL_IO_HPP

#include <string>
#include <vector>

namespace ppg {

/**
 * @brief 从文件读取信号数据到vector
 * @param filepath 输入文件路径
 * @param max_samples 最大读取样本数（0表示读取全部）
 * @return 信号数据数组
 */
std::vector<float> read_signal_from_file(
    const std::string& filepath, 
    int max_samples = 0
);

/**
 * @brief 保存信号到文件（vector版本）
 * @param signal 信号数据
 * @param filepath 输出文件路径
 * @param precision 小数精度（默认6位）
 */
void save_signal_to_file(
    const std::vector<float>& signal, 
    const std::string& filepath, 
    int precision = 6
);

} // namespace ppg

#endif // SIGNAL_IO_HPP

