
add_library(NvFlowLib SHARED IMPORTED)
set(NvFlowLib_IMPORTLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/win64")
set(NvFlowLib_IMPORTLIB_NAME "NvFlowLib$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:Debug>$<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:Release>_win64")
set_target_properties(NvFlowLib PROPERTIES
    IMPORTED_LOCATION_DEBUG             "${NvFlowLib_IMPORTLIB_DIR}/NvFlowLibDebug_win64.dll"
    IMPORTED_LOCATION_RELEASE           "${NvFlowLib_IMPORTLIB_DIR}/NvFlowLibRelease_win64.dll"
    IMPORTED_LOCATION_RELWITHDEBINFO    "${NvFlowLib_IMPORTLIB_DIR}/NvFlowLibDebug_win64.dll"
    IMPORTED_LOCATION_MINSIZEREL        "${NvFlowLib_IMPORTLIB_DIR}/NvFlowLibRelease_win64.dll"

    IMPORTED_IMPLIB_DEBUG               "${NvFlowLib_IMPORTLIB_DIR}/NvFlowLibDebug_win64.lib"
    IMPORTED_IMPLIB_RELEASE             "${NvFlowLib_IMPORTLIB_DIR}/NvFlowLibRelease_win64.lib"
    IMPORTED_IMPLIB_RELWITHDEBINFO      "${NvFlowLib_IMPORTLIB_DIR}/NvFlowLibDebug_win64.lib"
    IMPORTED_IMPLIB_MINSIZEREL          "${NvFlowLib_IMPORTLIB_DIR}/NvFlowLibRelease_win64.lib"

    INTERFACE_INCLUDE_DIRECTORIES       "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
set(NvFlowLib_PDB_FILE "${NvFlowLib_IMPORTLIB_DIR}/NvFlowLib_IMPORTLIB_NAME.pdb")

set(NvFlowLib_LINKED "")
if(NVFLOWDEMO_USE_REVERSED_LIB)
    set(NvFlowLib_LINKED NvFlow)
else()
    set(NvFlowLib_LINKED NvFlowLib)
endif()

add_library(SDL2 SHARED IMPORTED)
set(SDL2_IMPORTLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x64")
set(SDL2_IMPORTLIB_NAME "SDL2")
set_target_properties(SDL2 PROPERTIES
    IMPORTED_LOCATION               "${SDL2_IMPORTLIB_DIR}/${SDL2_IMPORTLIB_NAME}.dll"
    IMPORTED_IMPLIB                 "${SDL2_IMPORTLIB_DIR}/${SDL2_IMPORTLIB_NAME}.lib"
    INTERFACE_INCLUDE_DIRECTORIES   "${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/include"
)

add_library(SDL2main SHARED IMPORTED)
set_target_properties(SDL2main PROPERTIES
    IMPORTED_IMPLIB "${SDL2_IMPORTLIB_DIR}/SDL2main.lib"
)

set(DemoApp_POSTFIX
  "$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:Debug>$<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:Release>_win64")


# Shaders
nvflow_add_shader_object_headers(
    TARGET DemoAppShaders
    SHADER_PROFILE 5_0
    CONFIG_FILE "Shaders/Shaders.cfg"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Shaders/$<CONFIG>"
    INCLUDE_DIRECTORIES "include"
)
get_target_property(NvFlowDemo_SHADER_HEADER_FILES DemoAppShaders OBJECT_HEADER_FILES)
get_target_property(NvFlowDemo_SHADER_HEADER_OUTPUT_DIR  DemoAppShaders OBJECT_HEADER_OUTPUT_DIR)

file(GLOB DemoApp_CODE_GEN_SOURCES
    "DemoAppCodeGen/*.h"
    "DemoAppCodeGen/*.cpp"
)
add_executable(DemoAppCodeGen ${DemoApp_CODE_GEN_SOURCES})

file(GLOB DemoApp_D3D11_SOURCES
    "DemoAppD3D11/*.h"
    "DemoAppD3D11/*.cpp"
)
add_library(DemoAppD3D11 SHARED ${DemoApp_D3D11_SOURCES})
set_target_properties(DemoAppD3D11 PROPERTIES
    OUTPUT_NAME "DemoAppD3D11${DemoApp_POSTFIX}"
    INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${NvFlowDemo_SHADER_HEADER_OUTPUT_DIR}>"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)
target_link_libraries(DemoAppD3D11 ${NvFlowLib_LINKED} SDL2)
add_dependencies(DemoAppD3D11 DemoAppShaders)

file(GLOB DEMOAPP_D3D12_SOURCES
    "DemoAppD3D12/*.h"
    "DemoAppD3D12/*.cpp"
)
add_library(DemoAppD3D12 SHARED ${DEMOAPP_D3D12_SOURCES})
set_target_properties(DemoAppD3D12 PROPERTIES
    OUTPUT_NAME "DemoAppD3D12${DemoApp_POSTFIX}"
    INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${NvFlowDemo_SHADER_HEADER_OUTPUT_DIR}>"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)
target_link_libraries(DemoAppD3D12 ${NvFlowLib_LINKED} SDL2)
add_dependencies(DemoAppD3D12 DemoAppShaders)

add_executable(NvFlowDemoApp)
set_target_properties(NvFlowDemoApp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
)
get_target_property(NvFlowDemoApp_RUNTIME_OUTPUT_DIRECTORY NvFlowDemoApp RUNTIME_OUTPUT_DIRECTORY)

# data
file(GLOB NVFLOWDEMO_DATA_FILES "data/*.*")
copy_assets(
  TARGET NvFlowDemoData
  DESTINATION ${NvFlowDemoApp_RUNTIME_OUTPUT_DIRECTORY}/../../data
  SOURCES ${NVFLOWDEMO_DATA_FILES}
)

file(GLOB DemoApp_SOURCES
    "DemoApp/*.h"
    "DemoApp/*.cpp"
)

target_sources(NvFlowDemoApp PRIVATE ${DemoApp_SOURCES})
target_link_libraries(NvFlowDemoApp ${NvFlowLib_LINKED} SDL2 SDL2main "dxguid.lib")
add_dependencies(NvFlowDemoApp NvFlowDemoData DemoAppShaders)
target_include_directories(NvFlowDemoApp PRIVATE "$<BUILD_INTERFACE:${NvFlowDemo_SHADER_HEADER_OUTPUT_DIR}>")
target_compile_definitions(NvFlowDemoApp PRIVATE "DLL_SUFFIX=${DemoApp_POSTFIX}" "_CRT_SECURE_NO_WARNINGS")
if(WIN32 AND NOT ${CMAKE_CL_64})
  target_compile_definitions(NvFlowDemoApp PRIVATE "WINDOWS_IGNORE_PACKING_MISMATCH")
endif()

# Copy runtime depdendies
if(NVFLOWDEMO_USE_REVERSED_LIB)
    add_custom_command(
        TARGET NvFlowDemoApp
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "${NvFlowLib_IMPORTLIB_DIR}/${NvFlowLib_IMPORTLIB_NAME}.dll"
                "${NvFlowLib_IMPORTLIB_DIR}/${NvFlowLib_IMPORTLIB_NAME}.pdb"
                $<TARGET_FILE_DIR:NvFlowDemoApp>
        COMMAND_EXPAND_LISTS
    )
endif()

add_custom_command(
    TARGET NvFlowDemoApp
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${SDL2_IMPORTLIB_DIR}/${SDL2_IMPORTLIB_NAME}.dll"
        $<TARGET_FILE_DIR:NvFlowDemoApp>
)

