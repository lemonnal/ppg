#!/bin/bash
# DSPFilter 快速构建脚本
# 位置: /home/yogsothoth/桌面/workspace-ppg/DSPFilter/cmake_build.sh

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

# 设置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_DIR/build"
BUILD_TYPE="Release"  # 可选: Debug, Release, RelWithDebInfo, MinSizeRel
PARALLEL_JOBS=$(nproc)  # 使用所有可用CPU核心

# 可执行文件名称
EXECUTABLE="$BUILD_DIR/offline_main"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  PPG离线信号处理系统 - 构建脚本${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 解析命令行参数
CLEAN_BUILD=true  # 默认清理后重新构建
SKIP_RUN=false
BUILD_TYPE="Release"

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -n|--no-run)
            SKIP_RUN=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  -d, --debug      使用 Debug 模式构建（默认: Release）"
            echo "  -n, --no-run     构建但不运行程序"
            echo "  -h, --help       显示此帮助信息"
            echo ""
            echo "注意: 默认会清理构建目录后重新构建"
            echo ""
            echo "示例:"
            echo "  $0               # 清理并重新构建（Release模式）"
            echo "  $0 -d            # 清理并重新构建（Debug模式）"
            echo "  $0 -n            # 清理并构建但不运行"
            echo "  $0 -d -n         # Debug模式+不运行"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ 未知参数: $1${NC}"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 显示配置信息
echo -e "${YELLOW}📋 构建配置:${NC}"
echo -e "  项目目录: ${PROJECT_DIR}"
echo -e "  构建目录: ${BUILD_DIR}"
echo -e "  构建类型: ${BUILD_TYPE}"
echo -e "  并行任务: ${PARALLEL_JOBS} 个核心"
echo ""

# 清理旧的构建目录
if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}🧹 清理旧的构建目录...${NC}"
    rm -rf "$BUILD_DIR"
fi

# 创建构建目录
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}📂 创建构建目录...${NC}"
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

# 配置 CMake
echo -e "${YELLOW}🔧 配置项目 (CMake)...${NC}"
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ CMake 配置失败！${NC}"
    exit 1
fi

# 编译项目
echo -e "${YELLOW}🔨 编译项目...${NC}"
make -j$PARALLEL_JOBS

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ 编译失败！${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 编译成功！${NC}"
echo ""

# 显示可执行文件信息
if [ -f "$EXECUTABLE" ]; then
    echo -e "${BLUE}📦 可执行文件信息:${NC}"
    ls -lh "$EXECUTABLE" | awk '{print "  大小: " $5}'
    echo ""
fi

# 运行程序
if [ "$SKIP_RUN" = false ]; then
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}🚀 运行离线处理程序${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""

    "$EXECUTABLE"

    EXIT_CODE=$?
    echo ""
    echo -e "${BLUE}=========================================${NC}"

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ 程序运行完成！${NC}"
    else
        echo -e "${RED}❌ 程序异常退出 (退出码: $EXIT_CODE)${NC}"
        exit $EXIT_CODE
    fi
else
    echo -e "${YELLOW}⏭️  跳过程序运行${NC}"
    echo -e "${YELLOW}💡 手动运行: ${EXECUTABLE}${NC}"
fi

echo ""
exit 0
