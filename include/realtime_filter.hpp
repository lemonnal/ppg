#ifndef REALTIME_FILTER_HPP
#define REALTIME_FILTER_HPP

#include "DspFilters/Dsp.h"
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>

namespace ppg
{
    // ==================== RealtimeFilter 实现 ====================
    /**
     * @brief 实时IIR带通滤波器类（逐样本处理）
     *
     * 该类封装了Butterworth带通滤波器，支持逐样本输入和输出，
     * 适合嵌入式实时系统使用。
     */
    class RealtimeFilter
    {
    public:
        /**
         * @brief 构造函数
         * @param low_freq 低频截止频率 (Hz)
         * @param high_freq 高频截止频率 (Hz)
         * @param sample_rate 采样率 (Hz)
         * @param filter_order 滤波器阶数
         */
        RealtimeFilter(double low_freq, double high_freq,
                       double sample_rate, int filter_order = 3);

        /**
         * @brief 处理单个样本
         * @param input 输入样本值
         * @return 滤波后的样本值
         */
        float process_sample(float input);

        /**
         * @brief 重置滤波器状态
         */
        void reset();

        /**
         * @brief 使用初始值预热滤波器（减少瞬态响应）
         * @param initial_value 初始值（通常使用信号的均值）
         * @param num_samples 预热样本数
         */
        void warmup(float initial_value, int num_samples = 100);

    private:
        Dsp::SimpleFilter<Dsp::Butterworth::BandPass<6>, 1> filter_;
        double low_freq_;
        double high_freq_;
        double sample_rate_;
        int filter_order_;
    };

    /**
     * @brief 实时数据缓冲区类（滑动窗口）
     */
    class RealtimeBuffer
    {
    public:
        /**
         * @brief 构造函数
         * @param capacity 缓冲区容量（样本数）
         */
        RealtimeBuffer(size_t capacity);

        /**
         * @brief 添加新样本到缓冲区
         * @param sample 样本值
         */
        void push(float sample);

        /**
         * @brief 获取缓冲区中的所有数据
         * @return 数据向量
         */
        std::vector<float> get_data() const;

        /**
         * @brief 获取缓冲区大小
         * @return 当前样本数
         */
        size_t size() const { return buffer_.size(); }

        /**
         * @brief 检查缓冲区是否已满
         * @return true表示已满
         */
        bool is_full() const { return buffer_.size() == capacity_; }

        /**
         * @brief 清空缓冲区
         */
        void clear() { buffer_.clear(); }

        /**
         * @brief 获取最新的样本
         * @return 最新样本值
         */
        float get_latest() const
        {
            return buffer_.empty() ? 0.0f : buffer_.back();
        }

    private:
        std::deque<float> buffer_;
        size_t capacity_;
    };

    /**
     * @brief 整型数据缓冲区类（内存优化版本）
     *
     * 使用 int16_t 存储数据，相比 float 节省 50% 内存。
     * 适用于 ADC 采样数据和可以容忍精度损失的场景。
     */
    class RealtimeBufferInt16
    {
    public:
        /**
         * @brief 构造函数
         * @param capacity 缓冲区容量（样本数）
         */
        RealtimeBufferInt16(size_t capacity);

        /**
         * @brief 添加新样本到缓冲区
         * @param sample 整型样本值
         */
        void push(int16_t sample);

        /**
         * @brief 获取缓冲区中的所有数据（整型）
         * @return 整型数据向量
         */
        std::vector<int16_t> get_data_int() const;

        /**
         * @brief 获取缓冲区中的所有数据（转换为浮点）
         * @return 浮点数据向量
         */
        std::vector<float> get_data_float() const;

        /**
         * @brief 获取部分数据为浮点（减少拷贝开销）
         * @param start_idx 起始索引
         * @param length 要获取的样本数
         * @return 浮点数据向量
         */
        std::vector<float> get_data_float(size_t start_idx, size_t length) const;

        /**
         * @brief 获取缓冲区大小
         * @return 当前样本数
         */
        size_t size() const { return buffer_.size(); }

        /**
         * @brief 检查缓冲区是否已满
         * @return true表示已满
         */
        bool is_full() const { return buffer_.size() == capacity_; }

        /**
         * @brief 清空缓冲区
         */
        void clear() { buffer_.clear(); }

        /**
         * @brief 获取最新的样本
         * @return 最新样本值
         */
        int16_t get_latest() const
        {
            return buffer_.empty() ? 0 : buffer_.back();
        }

    private:
        std::deque<int16_t> buffer_;
        size_t capacity_;
    };

} // namespace ppg

#endif // REALTIME_FILTER_HPP
