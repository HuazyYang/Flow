
add_subdirectory(NvFlowShaders)

file(GLOB_RECURSE NvFlow_HEADERS "include/nvflow/*.h")
file(GLOB NvFlow_SOURCES "src/*.cpp")

add_library(NvFlow SHARED
    ${NvFlow_HEADERS}
    ${NvFlow_SOURCES}
)

get_target_property(NvFlowShaders_SHADER_HEADER_OUTPUT_DIR  NvFlowShaders OBJECT_HEADER_OUTPUT_DIR)

target_include_directories(NvFlow
    PUBLIC "include"
    PRIVATE "src/private" "$<BUILD_INTERFACE:${NvFlowShaders_SHADER_HEADER_OUTPUT_DIR}>"
)
set_target_properties(
    NvFlow PROPERTIES
    CXX_STANDARD 17
)
add_dependencies(NvFlow NvFlowShaders)

# add_executable(
#     TestVectorCached
#     src/TestVectorCached.cpp
#     src/Allocable.cpp
# )
# target_include_directories(TestVectorCached PRIVATE "src/private")

add_subdirectory(Tools)