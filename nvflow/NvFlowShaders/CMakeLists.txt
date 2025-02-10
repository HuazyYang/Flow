
find_program(FXC "fxc" REQUIRED)

function(nvflow_add_shaderlib)
    set(options "")
    set(oneValueArgs TARGET SHADER_PROFILE CONFIG_FILE OUTPUT_DIRECTORY)
    set(multiValueArgs INCLUDE_DIRECTORIES)

    cmake_parse_arguments("nvflow" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Sanity check
    if(NOT nvflow_TARGET OR NOT nvflow_CONFIG_FILE)
        message(FATAL_ERROR "nvflow_add_shaderlib: <TARGET> and <CONFIG_FILE> must be specified")
    endif()
    if(NOT nvflow_SHADER_PROFILE)
        set(nvflow_SHADER_PROFILE "5_0") # Default shader profile
    endif()
    if(NOT nvflow_OUTPUT_DIRECTORY)
        set(nvflow_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}) # Default output directory
    endif()

    # Make output directory if not present
    add_custom_command(
        OUTPUT ${nvflow_OUTPUT_DIRECTORY}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${nvflow_OUTPUT_DIRECTORY}
    )

    set(include_cmd_line "")
    set(include_file_list "")
    foreach(dir ${nvflow_INCLUDE_DIRECTORIES})
        string(CONCAT include_cmd "/I" ${dir})
        list(APPEND include_cmd_line ${include_cmd})
        file(GLOB GLOB_RECURSE  dir_file_list "${dir}/*.h")
        list(APPEND include_file_list ${dir_file_list})
    endforeach(dir ${nvflow_INCLUDE_DIRECTORIES})

    cmake_path(GET nvflow_CONFIG_FILE PARENT_PATH config_file_dir)
    if(NOT config_file_dir)
        set(config_file_dir ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    file(STRINGS ${nvflow_CONFIG_FILE} shader_cfg_lines REGEX "^[ \t\r\n]*[^#]+$")

    set(output_file_list "")

    foreach(cmd ${shader_cfg_lines})
        string(REGEX REPLACE "([ \t])\-([^ \t]+)" "\\1/\\2" cmd ${cmd})
        string(PREPEND cmd "fxc ")
        separate_arguments(cmd_line NATIVE_COMMAND ${cmd})

        list(APPEND cmd_line ${include_cmd_line})
        list(LENGTH cmd_line cmd_line_length)

        # Target shader profile
        list(FIND cmd_line "/T" target_index)
        MATH(EXPR target_index "${target_index}+1")
        if(${target_index} GREATER_EQUAL 1 AND ${target_index} LESS ${cmd_line_length})
            list(GET cmd_line ${target_index} target_profile)
            set(target_profile "${target_profile}_${nvflow_SHADER_PROFILE}")
            list(REMOVE_AT cmd_line ${target_index})
            list(INSERT cmd_line ${target_index} ${target_profile})
        endif()

        # Retrieve output file name
        set(output_filename "")
        list(FIND cmd_line "/E" target_index)
        MATH(EXPR target_index "${target_index}+1")
        if(${target_index} GREATER_EQUAL 1 AND ${target_index} LESS ${cmd_line_length})
            list(GET cmd_line ${target_index} output_filename)
        endif()
        if(output_filename STREQUAL "")
            message(FATAL_ERROR "nvflow_add_shaderlib: \"${cmd_line}\" does not have a entry function")
        endif()

        set(output_file_path "${nvflow_OUTPUT_DIRECTORY}/${output_filename}.h")
        list(APPEND output_file_list ${output_file_path})
        list(APPEND cmd_line "/Fh;${output_file_path}")

        list(APPEND cmd_line "$<IF:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,/Zi,/Zd>")
        list(APPEND cmd_line "$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:/Od>")

        list(GET cmd_line 1 entry_file_name)
        # Resolve entry file full path
        cmake_path(ABSOLUTE_PATH entry_file_name BASE_DIRECTORY ${config_file_dir} OUTPUT_VARIABLE entry_file_path)
        # Re-arrange entry file to last
        list(REMOVE_AT cmd_line 1)
        list(APPEND cmd_line ${entry_file_path})

        add_custom_command(OUTPUT ${output_file_path}
            COMMAND ${cmd_line}
            DEPENDS ${entry_file_path} ${include_file_list} ${nvflow_OUTPUT_DIRECTORY}
            COMMAND_EXPAND_LISTS
        )
    endforeach(cmd shader_cfg_lines)

    add_custom_target(
        ${nvflow_TARGET}
        ALL
        SOURCES ${output_file_list}
    )
endfunction(nvflow_add_shaderlib)

nvflow_add_shaderlib(
    TARGET NvFlowShaders
    SHADER_PROFILE 5_0
    CONFIG_FILE "Shaders.cfg"
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
    INCLUDE_DIRECTORIES "../NvFlow"
)
