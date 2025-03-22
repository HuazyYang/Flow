cmake_minimum_required(VERSION 3.21)

project(NvFlow VERSION 1.0.1)

# NvFlow config options
option(NVFLOW_USE_MICROSOFT_VLD         "NvFlow use Microsoft VLD library to detect user space moemory leakage" OFF)

# Demo config options
option(NVFLOWDEMO_USE_REVERSED_LIB      "NvFlow demo use reversed dll" ON)

include(cmake/ShaderTools.cmake)
include(cmake/CopyAsset.cmake)

add_subdirectory(nvflow)
add_subdirectory(demo)
