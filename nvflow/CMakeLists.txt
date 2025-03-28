# VLD
if(NVFLOW_USE_MICROSOFT_VLD)
    include(../cmake/FindVLD.cmake)
    set(NVFLOW__USE_MICROSOFT_VLD 1)
else()
    set(NVFLOW__USE_MICROSOFT_VLD 0)
endif()

add_subdirectory(NvFlowShaders)

file(GLOB_RECURSE NvFlow_HEADERS "include/nvflow/*.h")
file(GLOB NvFlow_SOURCES "src/*.cpp")

add_library(NvFlow SHARED
    ${NvFlow_HEADERS}
    ${NvFlow_SOURCES}
)

get_target_property(NvFlowShaders_SHADER_HEADER_OUTPUT_DIR  NvFlowShaders OBJECT_HEADER_OUTPUT_DIR)

target_include_directories(NvFlow
    PUBLIC "include/nvflow"
    PRIVATE "${NvFlowShaders_SHADER_HEADER_OUTPUT_DIR}" "include" "NvFlow" "src/private"
)
target_compile_definitions(NvFlow PRIVATE "NOMINMAX=1" "NVFLOW__USE_MICROSOFT_VLD=${NVFLOW__USE_MICROSOFT_VLD}")
set_target_properties(
    NvFlow PROPERTIES
    CXX_STANDARD 17
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)

target_link_libraries(NvFlow PRIVATE "dxguid.lib" "$<${NVFLOW__USE_MICROSOFT_VLD}:vld>")

add_dependencies(NvFlow NvFlowShaders)

add_subdirectory(Tools)