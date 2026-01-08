#!/bin/bash

# 彩色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 打印带颜色的分隔线
print_separator() {
    echo -e "${CYAN}================================================${NC}"
}

# 打印步骤信息
print_step() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} ${WHITE}$1${NC}"
}

# 打印成功信息
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# 打印错误信息
print_error() {
    echo -e "${RED}✗${NC} $1"
}

# 打印警告信息
print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 获取提交信息
COMMIT_MSG=${1:-"try auto commit"}

# 开始执行
clear
print_separator
echo -e "${PURPLE}        🚀 Git 自动提交脚本 🚀${NC}"
print_separator
echo ""

# 步骤1: 检查是否在git仓库中
print_step "检查 Git 仓库状态..."
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "当前目录不是 Git 仓库！"
    exit 1
fi
print_success "Git 仓库检查通过"
echo ""

# 步骤2: 显示当前状态
print_step "显示当前文件状态..."
git status --short
echo ""

# 步骤3: 添加所有文件
print_step "添加所有文件到暂存区..."
if git add .; then
    print_success "文件添加成功"
else
    print_error "文件添加失败！"
    exit 1
fi
echo ""

# 步骤4: 提交
print_step "提交更改: ${YELLOW}\"${COMMIT_MSG}\"${NC}"
if git commit -m "$COMMIT_MSG"; then
    print_success "提交成功"
else
    print_warning "提交失败（可能没有变更需要提交）"
fi
echo ""

# 步骤5: 推送到 origin
print_step "推送到远程仓库 (origin)..."
if git push origin; then
    print_success "推送到 origin 成功"
else
    print_error "推送到 origin 失败！"
    exit 1
fi
echo ""

# 步骤6: 推送到 backup
print_step "推送到备份仓库 (backup)..."
if git push backup; then
    print_success "推送到 backup 成功"
else
    print_error "推送到 backup 失败！"
    exit 1
fi
echo ""

# 完成
print_separator
echo -e "${GREEN}        ✨ 所有操作完成！✨${NC}"
print_separator
echo ""

