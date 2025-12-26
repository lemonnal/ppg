#!/bin/bash
# DSPFilters 快速启动脚本 - 使用 CMake 方法
# 位置: /home/yogsothoth/桌面/workspace-ppg/DSPFilters/my_example/build_cmake.sh

echo "=========================================="
echo "  DSPFilters 构建脚本 (CMake 方法)"
echo "=========================================="
echo ""

# 设置路径
PROJECT_DIR="/home/yogsothoth/桌面/workspace-ppg/DSPFilter"
BUILD_DIR="$PROJECT_DIR/build"

echo "📁 项目目录: $PROJECT_DIR"
echo "📁 构建目录: $BUILD_DIR"
echo ""

# 清理旧的构建目录
if [ -d "$BUILD_DIR" ]; then
    echo "🧹 清理旧的构建目录..."
    rm -rf "$BUILD_DIR"
fi

# 创建新的构建目录
echo "📂 创建构建目录..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "🔧 配置项目 (CMake)..."
cmake ..

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ CMake 配置失败！"
    exit 1
fi

echo ""
echo "🔨 编译项目 (Make)..."
make

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 编译成功！"
    echo ""
    echo "🚀 运行程序..."
    echo "=========================================="
    "$BUILD_DIR/main_example"
    echo "=========================================="
    echo ""
    echo "✅ 程序运行完成！"
else
    echo ""
    echo "❌ 编译失败！"
    echo "请检查错误信息并修复问题。"
    exit 1
fi
