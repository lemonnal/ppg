#!/bin/bash

# ECG离线QRS检测编译和运行脚本
echo "=========================================="
echo "ECG离线QRS检测 - Pan-Tomkins算法"
echo "=========================================="

# 进入build目录
cd "$(dirname "$0")/build" || exit 1

echo ""
echo "步骤1: 清理旧的构建文件..."
rm -rf CMakeFiles/ CMakeCache.txt cmake_install.cmake Makefile ecg_offline_main
echo "清理完成！"

echo ""
echo "步骤2: 运行CMake配置..."
cmake .. || exit 1

echo ""
echo "步骤3: 编译程序..."
make ecg_offline_main -j$(nproc) || exit 1

echo ""
echo "步骤4: 运行程序..."
echo "=========================================="
echo ""

# 检查可执行文件是否存在
if [ -f "./ecg_offline_main" ]; then
    # 运行程序（无需参数，已硬编码路径）
    ./ecg_offline_main
    
    echo ""
    echo "=========================================="
    echo "程序执行完成！"
    echo "=========================================="
else
    echo "错误: 可执行文件未生成"
    exit 1
fi

