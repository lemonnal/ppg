#include "ppg_analysis.hpp"
#include "find_peaks.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <cmath>

namespace ppg
{

    // ===================== 峰值检测函数 =====================

    void detect_peaks_and_valleys(
        const std::vector<float> &filtered_signal,
        double sample_rate,
        double min_time_interval,
        std::vector<int> &peaks,
        std::vector<int> &valleys,
        float &ac_component)
    {
        std::cout << "\n【峰值检测】" << std::endl;

        // 计算最小峰值间距
        int min_distance = static_cast<int>(sample_rate * min_time_interval);

        std::cout << "  采样率: " << sample_rate << " Hz" << std::endl;
        std::cout << "  最小峰值间距: " << min_distance << " 样本 ("
                  << min_time_interval << " 秒)" << std::endl;

        // 找峰值
        peaks = find_peaks(filtered_signal, min_distance);
        std::cout << "  检测到峰值数量: " << peaks.size() << std::endl;

        // 找谷值（信号取反）
        std::vector<float> inverted_signal = filtered_signal;
        for (float &val : inverted_signal)
        {
            val = -val;
        }
        valleys = find_peaks(inverted_signal, min_distance);
        std::cout << "  检测到谷值数量: " << valleys.size() << std::endl;

        // 打印前5个峰值
        if (!peaks.empty())
        {
            std::cout << "\n  前" << std::min(5, (int)peaks.size()) << "个峰值:" << std::endl;
            for (int i = 0; i < std::min(5, (int)peaks.size()); i++)
            {
                int idx = peaks[i];
                std::cout << "    峰值 " << (i + 1) << ": 位置=" << idx
                          << " (" << std::fixed << std::setprecision(3)
                          << (idx / sample_rate) << "s), 幅值="
                          << std::setprecision(2) << filtered_signal[idx] << std::endl;
            }
        }

        // 打印前5个谷值
        if (!valleys.empty())
        {
            std::cout << "\n  前" << std::min(5, (int)valleys.size()) << "个谷值:" << std::endl;
            for (int i = 0; i < std::min(5, (int)valleys.size()); i++)
            {
                int idx = valleys[i];
                std::cout << "    谷值 " << (i + 1) << ": 位置=" << idx
                          << " (" << std::fixed << std::setprecision(3)
                          << (idx / sample_rate) << "s), 幅值="
                          << std::setprecision(2) << filtered_signal[idx] << std::endl;
            }
        }

        // 计算平均AC分量（峰峰值）- 匹配峰值前后的谷值
        ac_component = 0.0f;
        if (!peaks.empty() && !valleys.empty())
        {
            float sum_ac = 0;
            int count = 0;

            for (int peak_idx : peaks)
            {
                // 找峰值前后最近的两个谷值
                int valley_before = -1;
                int valley_after = -1;

                for (int valley_idx : valleys)
                {
                    if (valley_idx < peak_idx)
                    {
                        if (valley_before == -1 || valley_idx > valley_before)
                            valley_before = valley_idx;
                    }
                    else if (valley_idx > peak_idx)
                    {
                        if (valley_after == -1 || valley_idx < valley_after)
                            valley_after = valley_idx;
                    }
                }

                // 使用前后谷值的平均值
                if (valley_before >= 0 && valley_after >= 0)
                {
                    float valley_avg = (filtered_signal[valley_before] + filtered_signal[valley_after]) / 2.0f;
                    float ac = filtered_signal[peak_idx] - valley_avg;
                    sum_ac += ac;
                    count++;
                }
                else if (valley_before >= 0)
                {
                    float ac = filtered_signal[peak_idx] - filtered_signal[valley_before];
                    sum_ac += ac;
                    count++;
                }
                else if (valley_after >= 0)
                {
                    float ac = filtered_signal[peak_idx] - filtered_signal[valley_after];
                    sum_ac += ac;
                    count++;
                }
            }

            if (count > 0)
            {
                ac_component = sum_ac / count;
                std::cout << "\n  平均AC分量（峰峰值）: " << std::fixed
                          << std::setprecision(2) << ac_component << std::endl;
            }
        }

        std::cout << "  峰值检测完成！" << std::endl;
    }

    // ===================== SpO2计算函数 =====================

    /**
     * @brief 基于双通道（红光+红外光）计算SpO2
     *
     * SpO2计算原理：
     * 1. 计算红光的AC/DC比值：R_red = AC_red / DC_red
     * 2. 计算红外光的AC/DC比值：R_ir = AC_ir / DC_ir
     * 3. 计算比值：R = R_red / R_ir
     * 4. 使用经验公式：SpO2 = (-3.7465271198e+01f * R^3 + 5.8403912586e+01f * R^2 + -3.7079378855e+01f * R + 1.0016136403e+02f)
     *
     * 物理原理：
     * - 氧合血红蛋白(HbO2)对红光吸收少，对红外光吸收多
     * - 脱氧血红蛋白(Hb)对红光吸收多，对红外光吸收少
     * - 通过红光/红外光的比值可以计算血氧饱和度
     */
    bool calculate_spo2_dual_channel(
        const std::vector<float> &red_input,
        const std::vector<float> &red_filtered,
        float red_ac,
        const std::vector<float> &ir_input,
        const std::vector<float> &ir_filtered,
        float ir_ac,
        float &spo2,
        float &ratio)
    {
        std::cout << "\n【SpO2估算 - 双通道方法】" << std::endl;
        std::cout << "  算法: 红光/红外光双通道AC/DC比值法（标准方法）" << std::endl;
        std::cout << "  原理: 利用氧合血红蛋白和脱氧血红蛋白的光吸收差异" << std::endl;

        // 计算红光的DC分量（原始信号均值）
        float red_dc = 0.0f;
        for (float val : red_input)
        {
            red_dc += val;
        }
        red_dc /= red_input.size();

        // 计算红外光的DC分量
        float ir_dc = 0.0f;
        for (float val : ir_input)
        {
            ir_dc += val;
        }
        ir_dc /= ir_input.size();

        std::cout << "\n  【红光通道 (660nm)】" << std::endl;
        std::cout << "    AC分量: " << std::fixed << std::setprecision(2) << red_ac << std::endl;
        std::cout << "    DC分量: " << red_dc << std::endl;

        std::cout << "\n  【红外光通道 (880nm)】" << std::endl;
        std::cout << "    AC分量: " << ir_ac << std::endl;
        std::cout << "    DC分量: " << ir_dc << std::endl;

        // 检查DC分量是否有效
        if (red_dc == 0 || ir_dc == 0)
        {
            std::cout << "\n  错误: DC分量为0，无法计算SpO2" << std::endl;
            return false;
        }

        // 计算归一化比值
        float red_ratio = red_ac / red_dc; // 红光的AC/DC
        float ir_ratio = ir_ac / ir_dc;    // 红外光的AC/DC

        std::cout << "\n  【归一化比值】" << std::endl;
        std::cout << "    红光 AC/DC: " << std::setprecision(6) << red_ratio << std::endl;
        std::cout << "    红外光 AC/DC: " << ir_ratio << std::endl;

        // 检查红外光比值是否有效
        if (ir_ratio == 0)
        {
            std::cout << "\n  错误: 红外光AC/DC比值为0，无法计算SpO2" << std::endl;
            return false;
        }

        // 计算R值（关键参数）
        ratio = red_ratio / ir_ratio;

        std::cout << "\n  【R值计算】" << std::endl;
        std::cout << "    R = (红光AC/DC) / (红外光AC/DC)" << std::endl;
        std::cout << "    R = " << std::setprecision(6) << ratio << std::endl;

        // 使用经验公式计算SpO2

        // 使用三次多项式计算SpO2
        spo2 = (-3.7465271198e+01f * std::pow(ratio, 3) +
                5.8403912586e+01f * std::pow(ratio, 2) +
                -3.7079378855e+01f * ratio +
                1.0016136403e+02f);

        // 限制在合理范围 (70-100%)
        spo2 = std::max(70.0f, std::min(100.0f, spo2));

        std::cout << "  估算SpO2: " << std::fixed << std::setprecision(1)
                  << std::setw(5) << spo2 << "%" << std::endl;

        return true;
    }

    // ===================== 心率计算函数 =====================

    bool calculate_heart_rate(
        const std::vector<int> &peaks,
        double sample_rate,
        float &heart_rate,
        float &hrv)
    {
        std::cout << "\n【心率计算】" << std::endl;
        std::cout << "  算法: 基于峰值间隔的时域方法（带异常值过滤）" << std::endl;

        if (peaks.size() < 2)
        {
            std::cout << "  错误: 峰值数量不足，无法计算心率" << std::endl;
            return false;
        }

        // 计算相邻峰值之间的间隔（单位：样本数）
        std::vector<float> intervals;
        std::vector<float> intervals_sec; // 间隔（秒）

        for (size_t i = 1; i < peaks.size(); i++)
        {
            int diff_samples = peaks[i] - peaks[i - 1];
            float diff_sec = diff_samples / static_cast<float>(sample_rate);
            intervals.push_back(static_cast<float>(diff_samples));
            intervals_sec.push_back(diff_sec);
        }

        // 第一步：计算初始中位数，用于异常值检测
        std::vector<float> intervals_sec_sorted = intervals_sec;
        std::sort(intervals_sec_sorted.begin(), intervals_sec_sorted.end());
        float median_interval = intervals_sec_sorted[intervals_sec_sorted.size() / 2];

        // 过滤异常值：保留在中位数±50%范围内的间隔
        std::vector<float> filtered_intervals_sec;
        int filtered_count = 0;
        for (float interval : intervals_sec)
        {
            float deviation = std::abs(interval - median_interval) / median_interval;
            if (deviation <= 0.5f) // 保留偏差≤50%的间隔
            {
                filtered_intervals_sec.push_back(interval);
            }
            else
            {
                filtered_count++;
            }
        }

        if (filtered_count > 0)
        {
            std::cout << "  ⚠ 检测到 " << filtered_count << " 个异常峰值间隔，已过滤" << std::endl;
        }

        // 如果过滤后间隔数不足，使用原始数据
        if (filtered_intervals_sec.size() < 2)
        {
            std::cout << "  警告: 过滤后间隔数不足，使用原始数据" << std::endl;
            filtered_intervals_sec = intervals_sec;
        }

        // 使用过滤后的间隔替换原始间隔
        intervals_sec = filtered_intervals_sec;

        // 计算平均间隔
        float sum_intervals = 0.0f;
        for (float interval : intervals_sec)
        {
            sum_intervals += interval;
        }
        float mean_interval = sum_intervals / intervals_sec.size();

        // 计算心率 (BPM = 60 / 平均间隔(秒))
        heart_rate = 60.0f / mean_interval;

        // 计算心率变异性 (HRV) - 使用间隔的标准差
        float variance = 0.0f;
        for (float interval : intervals_sec)
        {
            float diff = interval - mean_interval;
            variance += diff * diff;
        }
        variance /= intervals_sec.size();
        float std_dev_sec = std::sqrt(variance);
        hrv = std_dev_sec * 1000.0f; // 转换为毫秒

        std::cout << "\n  峰值数量: " << peaks.size() << std::endl;
        std::cout << "  有效间隔数: " << intervals.size() << std::endl;

        // 打印前5个间隔
        std::cout << "\n  前" << std::min(5, (int)intervals.size()) << "个峰值间隔:" << std::endl;
        for (int i = 0; i < std::min(5, (int)intervals.size()); i++)
        {
            std::cout << "    间隔 " << (i + 1) << ": " << std::fixed << std::setprecision(3)
                      << intervals_sec[i] << " s (" << std::setprecision(1)
                      << (60.0f / intervals_sec[i]) << " BPM)" << std::endl;
        }

        std::cout << "\n  平均RR间隔: " << std::setprecision(3) << mean_interval << " s" << std::endl;
        std::cout << "  平均RR间隔: " << std::setprecision(1) << (mean_interval * 1000.0f) << " ms" << std::endl;

        std::cout << "\n  ┌─────────────────────────────────┐" << std::endl;
        std::cout << "  │  估算心率: " << std::setprecision(1) << std::setw(5) << heart_rate
                  << " BPM         │" << std::endl;
        std::cout << "  └─────────────────────────────────┘" << std::endl;

        std::cout << "\n  心率变异性 (SDNN): " << std::setprecision(2) << hrv << " ms" << std::endl;

        // 心率范围评估
        std::cout << "\n  心率评估: ";
        if (heart_rate >= 60.0f && heart_rate <= 100.0f)
        {
            std::cout << "正常 ✓ (60-100 BPM)" << std::endl;
        }
        else if (heart_rate < 60.0f)
        {
            std::cout << "心动过缓 ⚠ (< 60 BPM)" << std::endl;
        }
        else
        {
            std::cout << "心动过速 ⚠ (> 100 BPM)" << std::endl;
        }

        // HRV评估
        std::cout << "  HRV评估: ";
        if (hrv >= 30.0f)
        {
            std::cout << "良好 ✓ (≥ 30 ms)" << std::endl;
        }
        else if (hrv >= 20.0f)
        {
            std::cout << "一般 ⚠ (20-30 ms)" << std::endl;
        }
        else
        {
            std::cout << "较低 ⚠ (< 20 ms)" << std::endl;
        }

        std::cout << "\n  注意: 此为估算值，仅供参考" << std::endl;

        return true;
    }

} // namespace ppg
