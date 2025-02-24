

nvflow_add_shader_object_headers(
    TARGET NvFlowShaders
    SHADER_PROFILE 5_0
    CONFIG_FILE "Shaders.cfg"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
    INCLUDE_DIRECTORIES "../NvFlow"
)
