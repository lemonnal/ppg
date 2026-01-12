#ifndef ECG_OFFLINE_HPP
#define ECG_OFFLINE_HPP

#include <vector>
#include <string>
#include <map>

namespace ecg {

/**
 * @brief 信号处理参数结构体
 * 对应 Python 中的 signal_params
 */
struct SignalParams {
    // 滤波参数
    float low;                          // 带通滤波器低频截止频率 (Hz)
    float high;                         // 带通滤波器高频截止频率 (Hz)
    int filter_order;                   // 滤波器阶数
    float original_weight;              // 原始信号权重
    float filtered_weight;              // 滤波后信号权重
    
    // QRS检测参数
    float integration_window_size;      // 积分窗口大小 (秒)
    float detection_window_size;        // 检测窗口大小 (秒)
    float overlap_window_size;          // 重叠窗口大小 (秒)
    float refractory_period;            // 不应期 (秒)
    float threshold_factor;             // 阈值系数
    
    // 默认构造函数 - MLII导联参数
    SignalParams()
        : low(5.0f)
        , high(15.0f)
        , filter_order(4)
        , original_weight(0.2f)
        , filtered_weight(0.8f)
        , integration_window_size(0.150f)
        , detection_window_size(5.0f)
        , overlap_window_size(4.0f)
        , refractory_period(0.2f)
        , threshold_factor(0.8f)
    {}
};

/**
 * @brief 心率统计信息结构体
 */
struct HeartRateStats {
    float mean_hr;      // 平均心率 (bpm)
    float std_hr;       // 心率标准差 (bpm)
    float min_hr;       // 最小心率 (bpm)
    float max_hr;       // 最大心率 (bpm)
    float median_hr;    // 中位数心率 (bpm)
    int num_beats;      // 心跳数
};

/**
 * @brief Pan-Tomkins算法的QRS波检测器 (离线版本)
 * 
 * 对应Python中的 PanTomkinsQRSDetectorOffline 类
 * 实现完整的Pan-Tomkins算法用于ECG信号的QRS波检测
 */
class PanTomkinsQRSDetectorOffline {
public:
    /**
     * @brief 构造函数
     * @param signal_name ECG导联名称 (如 "MLII", "V1", "V2" 等)
     * @param fs 采样频率 (Hz)，默认360Hz (MIT-BIH数据集)
     */
    PanTomkinsQRSDetectorOffline(const std::string& signal_name = "MLII", float fs = 360.0f);
    
    /**
     * @brief 析构函数
     */
    ~PanTomkinsQRSDetectorOffline();
    
    /**
     * @brief 检测QRS波峰值
     * @param signal_data 输入ECG信号
     * @return QRS波峰值位置索引
     */
    std::vector<int> detect_qrs_peaks(const std::vector<float>& signal_data);
    
    /**
     * @brief 计算RR间期
     * @param qrs_peaks R峰位置列表（可选，为空则使用内部的qrs_peaks_）
     * @param rr_intervals_sec 输出RR间期列表（单位：秒）
     * @param rr_intervals_samples 输出RR间期列表（单位：样本点）
     */
    void calculate_rr_intervals(
        const std::vector<int>& qrs_peaks,
        std::vector<float>& rr_intervals_sec,
        std::vector<int>& rr_intervals_samples
    );
    
    /**
     * @brief 计算瞬时心率
     * @param qrs_peaks R峰位置列表（可选，为空则使用内部的qrs_peaks_）
     * @param heart_rates 输出瞬时心率列表（单位：bpm）
     * @param time_stamps 输出对应的时间戳（单位：秒）
     */
    void calculate_instantaneous_heart_rate(
        const std::vector<int>& qrs_peaks,
        std::vector<float>& heart_rates,
        std::vector<float>& time_stamps
    );
    
    /**
     * @brief 计算平均心率
     * @param qrs_peaks R峰位置列表（可选，为空则使用内部的qrs_peaks_）
     * @param avg_heart_rate 输出平均心率（单位：bpm）
     * @param std_heart_rate 输出心率标准差（单位：bpm）
     */
    void calculate_average_heart_rate(
        const std::vector<int>& qrs_peaks,
        float& avg_heart_rate,
        float& std_heart_rate
    );
    
    /**
     * @brief 使用滑动窗口计算心率
     * @param window_size 滑动窗口大小（RR间期的数量）
     * @param qrs_peaks R峰位置列表（可选，为空则使用内部的qrs_peaks_）
     * @param sliding_hr 输出滑动窗口心率列表（单位：bpm）
     * @param time_stamps 输出对应的时间戳（单位：秒）
     */
    void calculate_sliding_window_heart_rate(
        int window_size,
        const std::vector<int>& qrs_peaks,
        std::vector<float>& sliding_hr,
        std::vector<float>& time_stamps
    );
    
    /**
     * @brief 获取心率统计信息
     * @param qrs_peaks R峰位置列表（可选，为空则使用内部的qrs_peaks_）
     * @return 包含各种心率统计信息的结构体
     */
    HeartRateStats get_heart_rate_statistics(const std::vector<int>& qrs_peaks);
    
    // 获取中间信号（用于调试和可视化）
    const std::vector<float>& get_filtered_signal() const { return filtered_signal_; }
    const std::vector<float>& get_differentiated_signal() const { return differentiated_signal_; }
    const std::vector<float>& get_squared_signal() const { return squared_signal_; }
    const std::vector<float>& get_integrated_signal() const { return integrated_signal_; }
    const std::vector<int>& get_qrs_peaks() const { return qrs_peaks_; }

private:
    /**
     * @brief 自适应带通滤波器
     * @param signal_data 输入ECG信号
     * @return 滤波后与原始信号加权组合的信号
     */
    std::vector<float> bandpass_filter(const std::vector<float>& signal_data);
    
    /**
     * @brief 优化的微分器 - 使用5点中心差分
     * @param signal_data 输入信号
     * @return 微分后的信号
     */
    std::vector<float> derivative(const std::vector<float>& signal_data);
    
    /**
     * @brief 平方函数
     * @param signal_data 输入信号
     * @return 平方后的信号
     */
    std::vector<float> squaring(const std::vector<float>& signal_data);
    
    /**
     * @brief 移动窗口积分器
     * @param signal_data 输入信号 (通常是微分平方后的信号)
     * @return 移动平均积分后的信号
     */
    std::vector<float> moving_window_integration(const std::vector<float>& signal_data);
    
    /**
     * @brief 滑动窗口阈值检测算法
     * @param signal_data 输入积分信号
     * @return 定位的QRS波峰值位置列表
     */
    std::vector<int> threshold_detection(const std::vector<float>& signal_data);
    
    /**
     * @brief 获取特定导联的参数
     * @param signal_name 导联名称
     * @return 参数结构体
     */
    SignalParams get_signal_params(const std::string& signal_name);

    // 成员变量
    float fs_;                              // 采样频率 (Hz)
    std::string signal_name_;               // 导联名称
    SignalParams params_;                   // 信号处理参数
    int refractory_period_samples_;         // 不应期（样本点数）
    
    // 中间处理信号
    std::vector<float> filtered_signal_;
    std::vector<float> differentiated_signal_;
    std::vector<float> squared_signal_;
    std::vector<float> integrated_signal_;
    std::vector<int> qrs_peaks_;
};

/**
 * @brief 从文件读取ECG信号
 * @param filename 文件名
 * @param max_samples 最大读取样本数（0表示读取全部）
 * @return 信号数据
 */
std::vector<float> read_ecg_signal(const std::string& filename, size_t max_samples = 0);

/**
 * @brief 保存信号到文件
 * @param signal 信号数据
 * @param filename 文件名
 * @return 是否成功
 */
bool save_signal_to_file(const std::vector<float>& signal, const std::string& filename);

} // namespace ecg

#endif // ECG_OFFLINE_HPP

