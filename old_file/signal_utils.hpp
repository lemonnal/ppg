#ifndef SIGNAL_UTILS_HPP
#define SIGNAL_UTILS_HPP

#include <vector>

namespace ppg {

/**
 * @brief 打印信号统计信息
 * @param input_signal 原始输入信号
 * @param filtered_signal 滤波后信号
 */
void print_signal_statistics(
    const std::vector<float>& input_signal,
    const std::vector<float>& filtered_signal
);

} // namespace ppg

#endif // SIGNAL_UTILS_HPP

