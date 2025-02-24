cmake_minimum_required(VERSION 3.21)

project(NvFlow VERSION 1.0.1)

include(cmake/ShaderTools.cmake)
include(cmake/CopyAsset.cmake)

add_subdirectory(nvflow)
add_subdirectory(demo)
