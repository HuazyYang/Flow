set(DemoApp_POSTFIX
  "$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:Debug>$<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:Release>_win64")
set(DemoApp_IMPORT_NVFLOWLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/win64")

add_library(NvFlowLib SHARED IMPORTED)
set_target_properties(NvFlowLib PROPERTIES
    IMPORTED_LOCATION_DEBUG             "${DemoApp_IMPORT_NVFLOWLIB_DIR}/NvFlowLibDebug_win64.dll"
    IMPORTED_LOCATION_RELEASE           "${DemoApp_IMPORT_NVFLOWLIB_DIR}/NvFlowLibRelease_win64.dll"
    IMPORTED_LOCATION_RELWITHDEBINFO    "${DemoApp_IMPORT_NVFLOWLIB_DIR}/NvFlowLibDebug_win64.dll"
    IMPORTED_LOCATION_MINSIZEREL        "${DemoApp_IMPORT_NVFLOWLIB_DIR}/NvFlowLibRelease_win64.dll"

    IMPORTED_IMPLIB_DEBUG               "${DemoApp_IMPORT_NVFLOWLIB_DIR}/NvFlowLibDebug_win64.lib"
    IMPORTED_IMPLIB_RELEASE             "${DemoApp_IMPORT_NVFLOWLIB_DIR}/NvFlowLibRelease_win64.lib"
    IMPORTED_IMPLIB_RELWITHDEBINFO      "${DemoApp_IMPORT_NVFLOWLIB_DIR}/NvFlowLibDebug_win64.lib"
    IMPORTED_IMPLIB_MINSIZEREL          "${DemoApp_IMPORT_NVFLOWLIB_DIR}/NvFlowLibRelease_win64.lib"

    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_library(SDL2 SHARED IMPORTED)
set(DemoApp_IMPORT_SDL2_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x64")
set_target_properties(SDL2 PROPERTIES
    IMPORTED_LOCATION   "${DemoApp_IMPORT_SDL2_DIR}/SDL2.dll"
    IMPORTED_IMPLIB     "${DemoApp_IMPORT_SDL2_DIR}/SDL2.lib"

    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/include"
)

add_library(SDL2main SHARED IMPORTED)
set_target_properties(SDL2main PROPERTIES
    IMPORTED_IMPLIB "${DemoApp_IMPORT_SDL2_DIR}/SDL2main.lib"
)

# Shaders
nvflow_add_shader_object_headers(
    TARGET DemoAppShaders
    SHADER_PROFILE 5_0
    CONFIG_FILE "demo/Shaders/Shaders.cfg"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Shaders/$<CONFIG>"
    INCLUDE_DIRECTORIES "include"
)
get_target_property(NvFlowDemo_SHADER_HEADER_FILES DemoAppShaders OBJECT_HEADER_FILES)
get_target_property(NvFlowDemo_SHADER_HEADER_OUTPUT_DIR  DemoAppShaders OBJECT_HEADER_OUTPUT_DIR)

file(GLOB DemoApp_CODE_GEN_SOURCES
    "demo/DemoAppCodeGen/*.h"
    "demo/DemoAppCodeGen/*.cpp"
)
add_executable(DemoAppCodeGen ${DemoApp_CODE_GEN_SOURCES})

file(GLOB DemoApp_D3D11_SOURCES
    "demo/DemoAppD3D11/*.h"
    "demo/DemoAppD3D11/*.cpp"
)
add_library(DemoAppD3D11 SHARED ${DemoApp_D3D11_SOURCES})
set_target_properties(DemoAppD3D11 PROPERTIES
    OUTPUT_NAME "DemoAppD3D11${DemoApp_POSTFIX}"
    INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${NvFlowDemo_SHADER_HEADER_OUTPUT_DIR}>"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)
target_link_libraries(DemoAppD3D11 NvFlowLib SDL2)
add_dependencies(DemoAppD3D11 DemoAppShaders)

file(GLOB DEMOAPP_D3D12_SOURCES
    "demo/DemoAppD3D12/*.h"
    "demo/DemoAppD3D12/*.cpp"
)
add_library(DemoAppD3D12 SHARED ${DEMOAPP_D3D12_SOURCES})
set_target_properties(DemoAppD3D12 PROPERTIES
    OUTPUT_NAME "DemoAppD3D12${DemoApp_POSTFIX}"
    INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${NvFlowDemo_SHADER_HEADER_OUTPUT_DIR}>"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)
target_link_libraries(DemoAppD3D12 NvFlowLib SDL2)
add_dependencies(DemoAppD3D12 DemoAppShaders)

add_executable(NvFlowDemoApp)
set_target_properties(NvFlowDemoApp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)
get_target_property(NvFlowDemoApp_RUNTIME_OUTPUT_DIRECTORY NvFlowDemoApp RUNTIME_OUTPUT_DIRECTORY)

copy_assets(
    TARGET NvFlowDemoAppRuntime
    DESTINATION ${NvFlowDemoApp_RUNTIME_OUTPUT_DIRECTORY}
    SOURCES
    "${DemoApp_IMPORT_NVFLOWLIB_DIR}/NvFlowLib${DemoApp_POSTFIX}.dll"
    "${DemoApp_IMPORT_NVFLOWLIB_DIR}/NvFlowLib${DemoApp_POSTFIX}.pdb"
    "${DemoApp_IMPORT_SDL2_DIR}/SDL2.dll"
)

# data
file(GLOB NVFLOWDEMO_DATA_FILES "data/*.*")
copy_assets(
  TARGET NvFlowDemoData
  DESTINATION ${NvFlowDemoApp_RUNTIME_OUTPUT_DIRECTORY}/../../data
  SOURCES ${NVFLOWDEMO_DATA_FILES}
)

file(GLOB DemoApp_SOURCES
    "demo/DemoApp/*.h"
    "demo/DemoApp/*.cpp"
)

target_sources(NvFlowDemoApp PRIVATE ${DemoApp_SOURCES})
target_link_libraries(NvFlowDemoApp NvFlowLib SDL2 SDL2main)
add_dependencies(NvFlowDemoApp NvFlowDemoAppRuntime NvFlowDemoData DemoAppShaders)
target_include_directories(NvFlowDemoApp PRIVATE "$<BUILD_INTERFACE:${NvFlowDemo_SHADER_HEADER_OUTPUT_DIR}>")
target_compile_definitions(NvFlowDemoApp PRIVATE "DLL_SUFFIX=${DemoApp_POSTFIX}" "_CRT_SECURE_NO_WARNINGS")

if(WIN32 AND NOT ${CMAKE_CL_64})
  target_compile_definitions(NvFlowDemoApp PRIVATE "WINDOWS_IGNORE_PACKING_MISMATCH")
endif()
