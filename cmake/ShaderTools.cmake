find_program(FXC_COMPILER "fxc" REQUIRED)

cmake_policy(SET CMP0121 NEW)

function(nvflow_add_shader_object_headers)
    set(options "")
    set(oneValueArgs TARGET SHADER_PROFILE CONFIG_FILE OUTPUT_DIRECTORY)
    set(multiValueArgs INCLUDE_DIRECTORIES)

    cmake_parse_arguments("nvflow" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Sanity check
    if(NOT nvflow_TARGET OR NOT nvflow_CONFIG_FILE)
        message(FATAL_ERROR "nvflow_add_shader_object_headers: <TARGET> and <CONFIG_FILE> must be specified")
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
        list(APPEND include_cmd_line "/I;${dir}")
        file(GLOB_RECURSE dir_file_list "${dir}/*.h")
        list(APPEND include_file_list ${dir_file_list})
    endforeach(dir ${nvflow_INCLUDE_DIRECTORIES})

    cmake_path(GET nvflow_CONFIG_FILE PARENT_PATH config_file_dir)
    if(NOT config_file_dir)
        set(config_file_dir ${CMAKE_CURRENT_SOURCE_DIR})
    else()
        cmake_path(ABSOLUTE_PATH config_file_dir OUTPUT_VARIABLE config_file_dir)
    endif()

    file(STRINGS ${nvflow_CONFIG_FILE} shader_cfg_lines REGEX "^[ \t\r\n]*[^#]+$")

    set(output_file_list "")
    set(entry_file_list "")

    foreach(cmd ${shader_cfg_lines})
        string(REGEX REPLACE "\-([^ \t]+)" "/\\1" cmd ${cmd})
        separate_arguments(cmd_line NATIVE_COMMAND ${cmd})

        # Resolve input file path and re-write command
        list(GET cmd_line 0 entry_file_name)
        # Retrieve input file extension
        cmake_path(GET entry_file_name EXTENSION LAST_ONLY entry_file_ext)
        # Resolve entry file full path
        cmake_path(ABSOLUTE_PATH entry_file_name BASE_DIRECTORY ${config_file_dir} OUTPUT_VARIABLE entry_file_path)

        # Include additional directories
        list(APPEND cmd_line ${include_cmd_line})
        list(LENGTH cmd_line cmd_line_length)

        # Re-target shader profile
        list(FIND cmd_line "/T" target_index)
        MATH(EXPR target_index "${target_index}+1")
        if(${target_index} GREATER_EQUAL 1 AND ${target_index} LESS ${cmd_line_length})
            list(GET cmd_line ${target_index} target_profile)
            set(target_profile "${target_profile}_${nvflow_SHADER_PROFILE}")
            list(REMOVE_AT cmd_line ${target_index})
            list(INSERT cmd_line ${target_index} ${target_profile})
        endif()

        # Specify output file path
        # Retrieve output file name
        set(output_filename "")
        list(FIND cmd_line "/E" target_index)
        MATH(EXPR target_index "${target_index}+1")
        if(${target_index} GREATER_EQUAL 1 AND ${target_index} LESS ${cmd_line_length})
            list(GET cmd_line ${target_index} output_filename)
        endif()
        if(output_filename STREQUAL "")
            message(FATAL_ERROR "nvflow_add_shader_object_headers: \"${cmd_line}\" does not have a entry function")
        endif()

        set(output_file_path "${nvflow_OUTPUT_DIRECTORY}/${output_filename}${entry_file_ext}.h")
        list(APPEND output_file_list ${output_file_path})
        list(APPEND cmd_line "/Fh;${output_file_path}")

        list(APPEND cmd_line "$<IF:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,/Zi,/Zd>")
        list(APPEND cmd_line "$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:/Od>")

        # Re-arrange entry file to last
        list(REMOVE_AT cmd_line 0)
        list(APPEND cmd_line ${entry_file_path})
        list(APPEND entry_file_list ${entry_file_path})

        add_custom_command(OUTPUT ${output_file_path}
            COMMAND ${FXC_COMPILER} ${cmd_line}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${entry_file_path} ${include_file_list} ${nvflow_OUTPUT_DIRECTORY}
            COMMAND_EXPAND_LISTS
        )
    endforeach(cmd shader_cfg_lines)

    add_custom_target(
        ${nvflow_TARGET}
        DEPENDS ${output_file_list}
        SOURCES ${include_file_list} ${entry_file_list}
    )

    # Set a header include directory path in parent scope
    define_property(TARGET PROPERTY OBJECT_HEADER_PUBLIC_DIR BRIEF_DOCS "Shader object header file include directory")
    define_property(TARGET PROPERTY OBJECT_HEADER_FILES BRIEF_DOCS "Shader object header file list")
    set_property(TARGET ${nvflow_TARGET}
        PROPERTY
        OBJECT_HEADER_OUTPUT_DIR ${nvflow_OUTPUT_DIRECTORY}
    )
    set_property(TARGET ${nvflow_TARGET}
        PROPERTY
        OBJECT_HEADER_FILES ${output_file_list}
    )
endfunction(nvflow_add_shader_object_headers)