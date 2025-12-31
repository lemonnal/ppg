# DSPFilter 构建指南

## 快速开始

### 标准构建并运行
```bash
./cmake_build.sh
```

### 常用命令

```bash
# 清理后重新构建（推荐在代码大改动后使用）
./cmake_build.sh -c

# Debug 模式构建（包含调试符号）
./cmake_build.sh -d

# 只构建不运行
./cmake_build.sh -n

# 清理 + Debug 模式 + 不运行
./cmake_build.sh -c -d -n

# 查看帮助信息
./cmake_build.sh -h
```

## 命令行参数

| 参数 | 简写 | 说明 |
|------|------|------|
| `--clean` | `-c` | 清理构建目录后重新构建 |
| `--debug` | `-d` | 使用 Debug 模式构建（默认: Release） |
| `--no-run` | `-n` | 构建但不运行程序 |
| `--help` | `-h` | 显示帮助信息 |

## 构建类型

- **Release**: 默认模式，启用优化，体积小，性能好
- **Debug**: 包含调试符号，无优化，适合使用 gdb 调试
- **RelWithDebInfo**: 优化 + 调试符号
- **MinSizeRel**: 最小体积优化

## 特性

✅ 自动并行编译（使用所有CPU核心）
✅ 彩色输出，易于阅读
✅ 错误时自动停止
✅ 显示可执行文件大小
✅ 灵活的命令行参数
