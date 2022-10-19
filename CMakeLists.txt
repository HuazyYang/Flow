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
    IMPORTED_LOCATION_DEBUG           ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibDebug_win64.dll
    IMPORTED_LOCATION_RELEASE         ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibRelease_win64.dll
    IMPORTED_LOCATION_RELWITHDEBINFO  ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibDebug_win64.dll
    IMPORTED_LOCATION_MINSIZEREL      ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibRelease_win64.dll

    IMPORTED_IMPLIB_DEBUG           ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibDebug_win64.lib
    IMPORTED_IMPLIB_RELEASE         ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibRelease_win64.lib
    IMPORTED_IMPLIB_RELWITHDEBINFO  ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibDebug_win64.lib
    IMPORTED_IMPLIB_MINSIZEREL      ${CMAKE_CURRENT_SOURCE_DIR}/lib/win64/NvFlowLibRelease_win64.lib
  )
  target_include_directories(NvFlowLib INTERFACE include)

  add_library(SDL2 SHARED IMPORTED)
  set_target_properties(
    SDL2
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x64/SDL2.dll
    IMPORTED_IMPLIB   ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x64/SDL2.lib
  )
  target_include_directories(SDL2 INTERFACE external/SDL2/include)

  add_library(SDL2main SHARED IMPORTED)
  set_target_properties(
    SDL2main
    PROPERTIES
    IMPORTED_IMPLIB   ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x64/SDL2main.lib
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
    IMPORTED_LOCATION_DEBUG           ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibDebug_win32.dll
    IMPORTED_LOCATION_RELEASE         ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibRelease_win32.dll
    IMPORTED_LOCATION_RELWITHDEBINFO  ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibDebug_win32.dll
    IMPORTED_LOCATION_MINSIZEREL      ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibRelease_win32.dll

    IMPORTED_IMPLIB_DEBUG           ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibDebug_win32.lib
    IMPORTED_IMPLIB_RELEASE         ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibRelease_win32.lib
    IMPORTED_IMPLIB_RELWITHDEBINFO  ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibDebug_win32.lib
    IMPORTED_IMPLIB_MINSIZEREL      ${CMAKE_CURRENT_SOURCE_DIR}/lib/win32/NvFlowLibRelease_win32.lib
  )
  target_include_directories(NvFlowLib INTERFACE include)

  add_library(SDL2 SHARED IMPORTED)
  set_target_properties(
    SDL2
    PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x86/SDL2.dll
    IMPORTED_IMPLIB   ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x86/SDL2.lib
  )
  target_include_directories(SDL2 INTERFACE external/SDL2/include)

  add_library(SDL2main SHARED IMPORTED)
  set_target_properties(
    SDL2main
    PROPERTIES
    IMPORTED_IMPLIB   ${CMAKE_CURRENT_SOURCE_DIR}/external/SDL2/lib/x86/SDL2main.lib
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

file(GLOB NVFLOWDEMO_APP_SOURCES "demo/DemoApp/*.h" "demo/DemoApp/*.cpp")
add_executable(NvFlowDemoApp ${NVFLOWDEMO_APP_SOURCES})
target_link_libraries(NvFlowDemoApp NvFlowLib SDL2 SDL2main)
add_dependencies(NvFlowDemoApp NvFlowDemoCopyExtraRuntime NvFlowDemoAppCopyData)
target_compile_definitions(NvFlowDemoApp PRIVATE "DLL_SUFFIX=${NVFLOWDEMO_POSTFIX}" "_CRT_SECURE_NO_WARNINGS")
if(WIN32 AND NOT ${CMAKE_CL_64})
  target_compile_definitions(NvFlowDemoApp PRIVATE "WINDOWS_IGNORE_PACKING_MISMATCH")
endif()

file(GLOB DEMOAPP_CODE_GEN_SOURCES "demo/DemoAppCodeGen/*.h" "demo/DemoAppCodeGen/*.cpp")
add_executable(DemoAppCodeGen ${DEMOAPP_CODE_GEN_SOURCES})

file(GLOB DEMOAPP_D3D11_SOURCES "demo/DemoAppD3D11/*.h" "demo/DemoAppD3D11/*.cpp")
add_library(DemoAppD3D11 SHARED ${DEMOAPP_D3D11_SOURCES})
set_target_properties(DemoAppD3D11 PROPERTIES OUTPUT_NAME "DemoAppD3D11${NVFLOWDEMO_POSTFIX}")
target_link_libraries(DemoAppD3D11 NvFlowLib SDL2)

file(GLOB DEMOAPP_D3D12_SOURCES "demo/DemoAppD3D12/*.h" "demo/DemoAppD3D12/*.cpp")
add_library(DemoAppD3D12 SHARED ${DEMOAPP_D3D12_SOURCES})
set_target_properties(DemoAppD3D12 PROPERTIES OUTPUT_NAME "DemoAppD3D12${NVFLOWDEMO_POSTFIX}")
target_link_libraries(DemoAppD3D12 NvFlowLib SDL2)


