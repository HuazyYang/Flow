cmake_minimum_required(VERSION 3.10)

include(cmake/CopyAsset.cmake)

project(NvFlowDemoApp)

set(CMAKE_DEBUG_POSTFIX "")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

set(NVFLOWDEMO_POSTFIX
  "$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:Debug>$<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:Release>_$<IF:${CMAKE_CL_64},win64,win32>")

if(${CMAKE_CL_64})
  add_library(NvFlowLib SHARED IMPORTED)
  set_target_properties(NvFlowLib PROPERTIES
    IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibDebug_win64.dll
    IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibRelease_win64.dll
    IMPORTED_LOCATION_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibDebug_win64.dll
    IMPORTED_LOCATION_MINSIZEREL ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibRelease_win64.dll

    IMPORTED_IMPLIB_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibDebug_win64.lib
    IMPORTED_IMPLIB_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibRelease_win64.lib
    IMPORTED_IMPLIB_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibDebug_win64.lib
    IMPORTED_IMPLIB_MINSIZEREL ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibRelease_win64.lib
  )
  target_include_directories(NvFlowLib INTERFACE include)

  add_library(SDL2 SHARED IMPORTED)
  set_target_properties(
    SDL2
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x64/SDL2.dll
    IMPORTED_IMPLIB ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x64/SDL2.lib
  )
  target_include_directories(SDL2 INTERFACE external/SDL2/include)

  add_library(SDL2main SHARED IMPORTED)
  set_target_properties(
    SDL2main
    PROPERTIES
    IMPORTED_IMPLIB ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x64/SDL2main.lib
  )

  copy_assets(
    "lib/win64/NvFlowLib${NVFLOWDEMO_POSTFIX}.dll;external/SDL2/lib/x64/SDL2.dll"
    ""
    NVFLOWDEMO_EXTRA_RUNTIME_FILES
  )
  add_custom_target(NvFlowDemoCopyExtraRuntime ALL DEPENDS ${NVFLOWDEMO_EXTRA_RUNTIME_FILES})
else()
  add_library(NvFlowLib SHARED IMPORTED)
  set_target_properties(NvFlowLib PROPERTIES
    IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibDebug_win32.dll
    IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibRelease_win32.dll
    IMPORTED_LOCATION_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibDebug_win32.dll
    IMPORTED_LOCATION_MINSIZEREL ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibRelease_win32.dll

    IMPORTED_IMPLIB_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibDebug_win32.lib
    IMPORTED_IMPLIB_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibRelease_win32.lib
    IMPORTED_IMPLIB_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibDebug_win32.lib
    IMPORTED_IMPLIB_MINSIZEREL ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibRelease_win32.lib
  )
  target_include_directories(NvFlowLib INTERFACE include)

  add_library(SDL2 SHARED IMPORTED)
  set_target_properties(
    SDL2
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x86/SDL2.dll
    IMPORTED_IMPLIB ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x86/SDL2.lib
  )
  target_include_directories(SDL2 INTERFACE external/SDL2/include)

  add_library(SDL2main SHARED IMPORTED)
  set_target_properties(
    SDL2main
    PROPERTIES
    IMPORTED_IMPLIB ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x86/SDL2main.lib
  )

  copy_assets(
    "lib/win32/NvFlowLib${NVFLOWDEMO_POSTFIX}.dll;external/SDL2/lib/x86/SDL2.dll"
    ""
    NVFLOWDEMO_EXTRA_RUNTIME_FILES
  )

  add_custom_target(NvFlowDemoCopyExtraRuntime ALL DEPENDS ${NVFLOWDEMO_EXTRA_RUNTIME_FILES})
endif()

# data
file(GLOB NVFLOWDEMO_DATA_FILES "data/*.*")
copy_assets("${NVFLOWDEMO_DATA_FILES}" "../../data" NVFLOWDEMO_DATA_COPIED_FILES)
add_custom_target(NvFlowDemoAppCopyData ALL DEPENDS ${NVFLOWDEMO_DATA_COPIED_FILES})

find_program(FXC_COMPILER fxc REQUIRED)
message("--- found fxc compiler: ${FXC_COMPILER}")
set(FXC_CFLAGS "$<IF:$<CONFIG:Debug>,/Od,/O2>" "$<$<CONFIG:Debug>:/Zi>" "/I:include")

# Shader object header output immediate directory
set(SHADER_OBJECT_HEADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/Shaders")
add_custom_command(OUTPUT ${SHADER_OBJECT_HEADER_OUTPUT_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OBJECT_HEADER_OUTPUT_DIR}
)

function(add_shader_header)
  set(oneValueArgs TARGET SHADER_TARGET SOURCE)
  cmake_parse_arguments(fxc "" "${oneValueArgs}" "" ${ARGN})

  cmake_path(GET fxc_SOURCE FILENAME source_name)
  cmake_path(GET fxc_SOURCE STEM source_fname)
  add_custom_command(
    OUTPUT "${SHADER_OBJECT_HEADER_OUTPUT_DIR}/${source_name}.h"
    COMMAND ${FXC_COMPILER} ${FXC_CFLAGS} -T ${fxc_SHADER_TARGET} -E ${source_fname} /Fh "${SHADER_OBJECT_HEADER_OUTPUT_DIR}/${source_name}.h" "${fxc_SOURCE}"
    DEPENDS ${fxc_SOURCE} ${SHADER_OBJECT_HEADER_OUTPUT_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_sources(${fxc_TARGET} PRIVATE "${SHADER_OBJECT_HEADER_OUTPUT_DIR}/${source_name}.h")
  target_include_directories(${fxc_TARGET} PRIVATE "$<BUILD_INTERFACE:${SHADER_OBJECT_HEADER_OUTPUT_DIR}>")
endfunction(add_shader_header)

file(GLOB NVFLOWDEMO_APP_SOURCES
  "demo/DemoApp/*.h"
  "demo/DemoApp/*.cpp"
  "${SHADER_OBJECT_HEADER_OUTPUT_DIR}/customEmitAllocCS.hsl.h"
  "${SHADER_OBJECT_HEADER_OUTPUT_DIR}/customEmitEmit2CS.hsl.h"
  "${SHADER_OBJECT_HEADER_OUTPUT_DIR}/customEmitEmitCS.hsl.h"
  "${SHADER_OBJECT_HEADER_OUTPUT_DIR}/customLightingCS.hsl.h"
)
add_executable(NvFlowDemoApp ${NVFLOWDEMO_APP_SOURCES})
add_shader_header(TARGET NvFlowDemoApp SHADER_TARGET cs_5_0 SOURCE "demo/Shaders/customEmitAllocCS.hlsl")
add_shader_header(TARGET NvFlowDemoApp SHADER_TARGET cs_5_0 SOURCE "demo/Shaders/customEmitEmit2CS.hlsl")
add_shader_header(TARGET NvFlowDemoApp SHADER_TARGET cs_5_0 SOURCE "demo/Shaders/customEmitEmitCS.hlsl")
add_shader_header(TARGET NvFlowDemoApp SHADER_TARGET cs_5_0 SOURCE "demo/Shaders/customLightingCS.hlsl")
target_link_libraries(NvFlowDemoApp NvFlowLib SDL2 SDL2main)
add_dependencies(NvFlowDemoApp NvFlowDemoCopyExtraRuntime NvFlowDemoAppCopyData)
target_include_directories(NvFlowDemoApp PRIVATE "$<BUILD_INTERFACE:${SHADER_OBJECT_HEADER_OUTPUT_DIR}>")
target_compile_definitions(NvFlowDemoApp PRIVATE "DLL_SUFFIX=${NVFLOWDEMO_POSTFIX}" "_CRT_SECURE_NO_WARNINGS")

if(WIN32 AND NOT ${CMAKE_CL_64})
  target_compile_definitions(NvFlowDemoApp PRIVATE "WINDOWS_IGNORE_PACKING_MISMATCH")
endif()

file(GLOB DEMOAPP_CODE_GEN_SOURCES "demo/DemoAppCodeGen/*.h" "demo/DemoAppCodeGen/*.cpp")
add_executable(DemoAppCodeGen ${DEMOAPP_CODE_GEN_SOURCES})

file(GLOB DEMOAPP_D3D11_SOURCES
  "demo/DemoAppD3D11/*.h"
  "demo/DemoAppD3D11/*.cpp"
)
add_library(DemoAppD3D11 SHARED ${DEMOAPP_D3D11_SOURCES})
add_shader_header(TARGET DemoAppD3D11 SHADER_TARGET vs_5_0 SOURCE "demo/Shaders/imguiVS.hlsl")
add_shader_header(TARGET DemoAppD3D11 SHADER_TARGET ps_5_0 SOURCE "demo/Shaders/imguiPS.hlsl")
add_shader_header(TARGET DemoAppD3D11 SHADER_TARGET vs_5_0 SOURCE "demo/Shaders/meshVS.hlsl")
add_shader_header(TARGET DemoAppD3D11 SHADER_TARGET ps_5_0 SOURCE "demo/Shaders/meshPS.hlsl")
set_target_properties(DemoAppD3D11 PROPERTIES OUTPUT_NAME "DemoAppD3D11${NVFLOWDEMO_POSTFIX}" INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${SHADER_OBJECT_HEADER_OUTPUT_DIR}>")
target_link_libraries(DemoAppD3D11 NvFlowLib SDL2)

file(GLOB DEMOAPP_D3D12_SOURCES
  "demo/DemoAppD3D12/*.h"
  "demo/DemoAppD3D12/*.cpp"
)
add_library(DemoAppD3D12 SHARED ${DEMOAPP_D3D12_SOURCES})
add_shader_header(TARGET DemoAppD3D12 SHADER_TARGET vs_5_0 SOURCE "demo/Shaders/imguiVS.hlsl")
add_shader_header(TARGET DemoAppD3D12 SHADER_TARGET ps_5_0 SOURCE "demo/Shaders/imguiPS.hlsl")
add_shader_header(TARGET DemoAppD3D12 SHADER_TARGET vs_5_0 SOURCE "demo/Shaders/meshVS.hlsl")
add_shader_header(TARGET DemoAppD3D12 SHADER_TARGET ps_5_0 SOURCE "demo/Shaders/meshPS.hlsl")
set_target_properties(DemoAppD3D12 PROPERTIES OUTPUT_NAME "DemoAppD3D12${NVFLOWDEMO_POSTFIX}" INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${SHADER_OBJECT_HEADER_OUTPUT_DIR}>")
target_link_libraries(DemoAppD3D12 NvFlowLib SDL2)
