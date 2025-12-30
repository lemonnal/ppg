cmake_minimum_required(VERSION 3.10)
project(DSPFiltersExample)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加 DSPFilters 库
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DSPFilters ${CMAKE_CURRENT_BINARY_DIR}/DSPFilters_build)

# 禁用 DSPFilters 库的警告
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(DSPFilters PRIVATE
        -Wno-unused-parameter
        -Wno-ignored-qualifiers
    )
elseif(MSVC)
    target_compile_options(DSPFilters PRIVATE
        /wd4100
        /wd4189
    )
endif()


# 创建示例可执行文件
add_executable(main_example main.cpp)

# 链接 DSPFilters 库
target_link_libraries(main_example PRIVATE DSPFilters)

# 设置包含目录
target_include_directories(main_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/DSPFilters/include
)

# 禁用来自第三方库头文件的警告
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(main_example PRIVATE
        -Wno-unused-parameter
        -Wno-ignored-qualifiers
        -Wall
        -Wextra
    )
endif()

# 如果是 MSVC，启用更多警告
if(MSVC)
    target_compile_options(main_example PRIVATE /W4)
endif()
