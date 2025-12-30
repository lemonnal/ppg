#ifndef FIND_PEAKS_H
#define FIND_PEAKS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

/**
 * @brief C++ implementation of scipy.signal.find_peaks
 * 
 * 仿照scipy.signal.find_peaks实现
 * 主要用于PPG信号的峰值检测
 */

// =====================================================================
// 核心数据结构
// =====================================================================

struct PeakProperties {
    std::vector<int> peak_heights;      // 峰值高度
    std::vector<float> prominences;     // 显著性
    std::vector<int> left_bases;        // 左侧基线位置
    std::vector<int> right_bases;       // 右侧基线位置
};

// =====================================================================
// 步骤1: 找到所有局部最大值 (_local_maxima_1d)
// =====================================================================

/**
 * @brief 找到信号中的所有局部最大值点
 * 
 * 对应scipy中的 _local_maxima_1d(x)
 * 
 * 局部最大值定义：x[i-1] < x[i] >= x[i+1]
 */
std::vector<int> find_local_maxima(const std::vector<float>& signal) {
    std::vector<int> peaks;
    
    if (signal.size() < 3) {
        return peaks;  // 信号太短，无法找峰值
    }
    
    // 扫描信号，找局部最大值
    for (size_t i = 1; i < signal.size() - 1; i++) {
        // 核心条件：左边 < 当前 且 当前 >= 右边
        if (signal[i - 1] < signal[i] && signal[i] >= signal[i + 1]) {
            peaks.push_back(i);
        }
    }
    
    return peaks;
}

// =====================================================================
// 步骤2: 应用distance约束 (_select_by_peak_distance)
// =====================================================================

/**
 * @brief 根据distance约束过滤峰值
 * 
 * 对应scipy中的 _select_by_peak_distance(peaks, priority, distance)
 * 
 * 算法：
 * 1. 按优先级（峰值高度）排序
 * 2. 贪心选择：保留优先级高的峰值，删除距离太近的峰值
 * 
 * @param peaks 峰值索引
 * @param signal 原始信号
 * @param distance 最小间距
 * @return 过滤后的峰值索引
 */
std::vector<int> filter_peaks_by_distance(
    const std::vector<int>& peaks,
    const std::vector<float>& signal,
    int distance
) {
    if (distance <= 0 || peaks.empty()) {
        return peaks;
    }
    
    // 创建(索引, 高度, 原始位置)的三元组
    struct PeakInfo {
        int index;           // 在peaks数组中的位置
        float height;        // 峰值高度
        int position;        // 在信号中的位置
    };
    
    std::vector<PeakInfo> peak_infos;
    for (size_t i = 0; i < peaks.size(); i++) {
        peak_infos.push_back({
            static_cast<int>(i),
            signal[peaks[i]],
            peaks[i]
        });
    }
    
    // 按高度降序排序（优先级：高度越大越优先保留）
    std::sort(peak_infos.begin(), peak_infos.end(),
              [](const PeakInfo& a, const PeakInfo& b) {
                  return a.height > b.height;
              });
    
    // 贪心算法：从高到低选择峰值
    std::vector<bool> keep(peaks.size(), true);  // 标记哪些峰值保留
    
    for (size_t i = 0; i < peak_infos.size(); i++) {
        if (!keep[peak_infos[i].index]) {
            continue;  // 已被删除
        }
        
        int current_pos = peak_infos[i].position;
        
        // 删除所有距离太近且优先级更低的峰值
        for (size_t j = i + 1; j < peak_infos.size(); j++) {
            int other_pos = peak_infos[j].position;
            
            if (std::abs(current_pos - other_pos) < distance) {
                keep[peak_infos[j].index] = false;  // 删除优先级低的
            }
        }
    }
    
    // 收集保留的峰值（按原始顺序）
    std::vector<int> filtered_peaks;
    for (size_t i = 0; i < peaks.size(); i++) {
        if (keep[i]) {
            filtered_peaks.push_back(peaks[i]);
        }
    }
    
    return filtered_peaks;
}

// =====================================================================
// 步骤3: 应用height约束
// =====================================================================

/**
 * @brief 根据height约束过滤峰值
 * 
 * @param peaks 峰值索引
 * @param signal 原始信号
 * @param min_height 最小高度（负无穷表示无限制）
 * @param max_height 最大高度（正无穷表示无限制）
 * @return 过滤后的峰值索引
 */
std::vector<int> filter_peaks_by_height(
    const std::vector<int>& peaks,
    const std::vector<float>& signal,
    float min_height = -std::numeric_limits<float>::infinity(),
    float max_height = std::numeric_limits<float>::infinity()
) {
    std::vector<int> filtered_peaks;
    
    for (int peak_idx : peaks) {
        float height = signal[peak_idx];
        
        if (height >= min_height && height <= max_height) {
            filtered_peaks.push_back(peak_idx);
        }
    }
    
    return filtered_peaks;
}

// =====================================================================
// 步骤4: 计算prominence（显著性）
// =====================================================================

/**
 * @brief 计算峰值的显著性（prominence）
 * 
 * Prominence定义：峰值相对于周围最低点的高度
 * 
 * @param signal 原始信号
 * @param peak_idx 峰值索引
 * @param left_base 左侧基线位置（输出）
 * @param right_base 右侧基线位置（输出）
 * @return prominence值
 */
float calculate_prominence(
    const std::vector<float>& signal,
    int peak_idx,
    int& left_base,
    int& right_base
) {
    float peak_height = signal[peak_idx];
    
    // 向左找最低点
    float left_min = peak_height;
    left_base = 0;
    for (int i = peak_idx - 1; i >= 0; i--) {
        if (signal[i] < left_min) {
            left_min = signal[i];
            left_base = i;
        }
        // 如果遇到更高的峰，停止
        if (signal[i] > peak_height) {
            break;
        }
    }
    
    // 向右找最低点
    float right_min = peak_height;
    right_base = signal.size() - 1;
    for (size_t i = peak_idx + 1; i < signal.size(); i++) {
        if (signal[i] < right_min) {
            right_min = signal[i];
            right_base = i;
        }
        // 如果遇到更高的峰，停止
        if (signal[i] > peak_height) {
            break;
        }
    }
    
    // Prominence = 峰值 - 两侧最低点中的较高者
    float base_height = std::max(left_min, right_min);
    return peak_height - base_height;
}

/**
 * @brief 根据prominence约束过滤峰值
 */
std::vector<int> filter_peaks_by_prominence(
    const std::vector<int>& peaks,
    const std::vector<float>& signal,
    float min_prominence = 0.0f
) {
    std::vector<int> filtered_peaks;
    
    for (int peak_idx : peaks) {
        int left_base, right_base;
        float prom = calculate_prominence(signal, peak_idx, left_base, right_base);
        
        if (prom >= min_prominence) {
            filtered_peaks.push_back(peak_idx);
        }
    }
    
    return filtered_peaks;
}

// =====================================================================
// 主函数：find_peaks（简化版，重点实现distance）
// =====================================================================

/**
 * @brief C++版本的scipy.signal.find_peaks
 * 
 * 用法示例：
 *   std::vector<float> signal = {1, 3, 5, 4, 2, 3, 6, 4, 1};
 *   auto peaks = find_peaks(signal, 2);  // distance=2
 *   // peaks = {2, 6}
 * 
 * @param signal 输入信号
 * @param distance 峰值最小间距（样本数）
 * @param min_height 峰值最小高度（可选）
 * @param min_prominence 峰值最小显著性（可选）
 * @return 峰值索引数组
 */
std::vector<int> find_peaks(
    const std::vector<float>& signal,
    int distance = 0,
    float min_height = -std::numeric_limits<float>::infinity(),
    float min_prominence = -1.0f  // 负数表示不使用prominence约束
) {
    // 步骤1：找到所有局部最大值
    std::vector<int> peaks = find_local_maxima(signal);
    
    if (peaks.empty()) {
        return peaks;
    }
    
    // 步骤2：应用height约束
    if (std::isfinite(min_height)) {
        peaks = filter_peaks_by_height(peaks, signal, min_height);
    }
    
    // 步骤3：应用distance约束（核心！）
    if (distance > 0) {
        peaks = filter_peaks_by_distance(peaks, signal, distance);
    }
    
    // 步骤4：应用prominence约束
    if (min_prominence >= 0.0f) {
        peaks = filter_peaks_by_prominence(peaks, signal, min_prominence);
    }
    
    return peaks;
}

// =====================================================================
// 完整版：返回peaks和properties
// =====================================================================

/**
 * @brief 完整版find_peaks，返回peaks和properties
 * 
 * @param signal 输入信号
 * @param peaks 输出：峰值索引
 * @param properties 输出：峰值属性
 * @param distance 峰值最小间距
 * @param min_height 最小高度
 * @param min_prominence 最小显著性
 */
void find_peaks_with_properties(
    const std::vector<float>& signal,
    std::vector<int>& peaks,
    PeakProperties& properties,
    int distance = 0,
    float min_height = -std::numeric_limits<float>::infinity(),
    float min_prominence = -1.0f
) {
    // 找峰值
    peaks = find_peaks(signal, distance, min_height, min_prominence);
    
    // 计算属性
    properties.peak_heights.clear();
    properties.prominences.clear();
    properties.left_bases.clear();
    properties.right_bases.clear();
    
    for (int peak_idx : peaks) {
        // 高度
        properties.peak_heights.push_back(signal[peak_idx]);
        
        // 显著性
        int left_base, right_base;
        float prom = calculate_prominence(signal, peak_idx, left_base, right_base);
        properties.prominences.push_back(prom);
        properties.left_bases.push_back(left_base);
        properties.right_bases.push_back(right_base);
    }
}

#endif // FIND_PEAKS_H

