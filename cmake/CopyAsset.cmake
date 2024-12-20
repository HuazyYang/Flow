# copy asset file from source directory to binary directory.
function(copy_assets asset_files dir_name copied_files)
foreach(asset ${asset_files})
  #message("asset: ${asset}")
  get_filename_component(file_name ${asset} NAME)
  get_filename_component(full_path ${asset} ABSOLUTE)
  set(output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${dir_name})
  set(output_file ${output_dir}/${file_name})
  set(${copied_files} ${${copied_files}} ${output_file})
  set(${copied_files} ${${copied_files}} PARENT_SCOPE)
  set_source_files_properties(${asset} PROPERTIES HEADER_FILE_ONLY TRUE)
  if (WIN32)
    add_custom_command(
      OUTPUT ${output_file}
      #COMMAND mklink \"${output_file}\" \"${full_path}\"
      COMMAND xcopy \"${full_path}\" \"${output_file}*\" /Y /Q /F
      DEPENDS ${full_path}
    )
  else()
    add_custom_command(
      OUTPUT ${output_file}
      COMMAND mkdir --parents ${output_dir} && cp --force --link \"${full_path}\" \"${output_file}\"
      DEPENDS ${full_path}
    )
  endif()
endforeach()
endfunction()