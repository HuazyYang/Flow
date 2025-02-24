

add_executable(
    RecoverHLSL
    RecoverHLSL.cpp
    dxbc_spdb.cpp
    dxbc_spdb.h
)
target_link_libraries(RecoverHLSL PRIVATE DXCompiler D3DCompiler)
set_target_properties(RecoverHLSL PROPERTIES CXX_STANDARD 17)