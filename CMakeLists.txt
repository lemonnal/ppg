cmake_minimum_required(VERSION 3.10)
project(PPG_Signal_Processing)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

################################################################################
# 第三方库配置
################################################################################

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

# ################################################################################
# # 离线信号处理程序 (offline_main)
# ################################################################################

# # 创建离线处理可执行文件
# add_executable(offline_main 
#     offline_main.cpp
#     src/signal_io.cpp
#     src/ppg_filters.cpp
#     src/ppg_analysis.cpp
#     src/signal_utils.cpp    
#     src/find_peaks.cpp
# )

# # 链接 DSPFilters 库
# target_link_libraries(offline_main PRIVATE DSPFilters)

# # 设置包含目录
# target_include_directories(offline_main PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/DSPFilters/include
#     ${CMAKE_CURRENT_SOURCE_DIR}
# )

# # 禁用来自第三方库头文件的警告
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     target_compile_options(offline_main PRIVATE
#         -Wno-unused-parameter
#         -Wno-ignored-qualifiers
#         -Wall
#         -Wextra
#     )
# endif()

# # 如果是 MSVC，启用更多警告
# if(MSVC)
#     target_compile_options(offline_main PRIVATE /W4)
# endif()

# ###############################################################################
# 实时信号处理程序 (realtime_main)
# ###############################################################################

# 创建实时处理可执行文件
add_executable(realtime_main 
    realtime_main.cpp
    src/ppg_filters.cpp
    src/ppg_analysis.cpp  
    src/find_peaks.cpp
    src/realtime_filter.cpp
)

# 链接 DSPFilters 库
target_link_libraries(realtime_main PRIVATE DSPFilters)

# 设置包含目录
target_include_directories(realtime_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/DSPFilters/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 禁用来自第三方库头文件的警告
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(realtime_main PRIVATE
        -Wno-unused-parameter
        -Wno-ignored-qualifiers
        -Wall
        -Wextra
    )
endif()

# 如果是 MSVC，启用更多警告
if(MSVC)
    target_compile_options(realtime_main PRIVATE /W4)
endif()

# ################################################################################
# # ECG离线QRS检测程序 (ecg_offline_main) - 新增
# ################################################################################

# # 创建ECG QRS检测可执行文件
# add_executable(ecg_offline_main 
#     ecg_offline_main.cpp
#     src/ecg_offline.cpp
# )

# # 链接 DSPFilters 库
# target_link_libraries(ecg_offline_main PRIVATE DSPFilters)

# # 设置包含目录
# target_include_directories(ecg_offline_main PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/DSPFilters/include
#     ${CMAKE_CURRENT_SOURCE_DIR}
# )

# # 禁用来自第三方库头文件的警告
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     target_compile_options(ecg_offline_main PRIVATE
#         -Wno-unused-parameter
#         -Wno-ignored-qualifiers
#         -Wall
#         -Wextra
#     )
# endif()

# # 如果是 MSVC，启用更多警告
# if(MSVC)
#     target_compile_options(ecg_offline_main PRIVATE /W4)
# endif()