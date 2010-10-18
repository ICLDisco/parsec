#
# Sources depending on Precision 
#
set(PRECISION d s) # z c d s                                         
set(PRECISION_CODEGEN ${CMAKE_SOURCE_DIR}/tools/precision_generator/codegen.py )

set(PRECISION_STUBS zmatrix.c)

#  precision generation 
set(generated_files)
foreach(_prec ${PRECISION})
  set(generated_files_${_prec})
  if( _prec MATCHES "z" )
    list(APPEND generated_files_${_prec} "${PRECISION_STUBS}")
  else()
    foreach(_file ${PRECISION_STUBS})
      string(REGEX REPLACE "^z" "${_prec}" file_tro ${_file})
      message(STATUS "gen ${file_tro} from ${_file} with ${PRECISION_CODEGEN}")
      add_custom_command(
        OUTPUT ${file_tro}
        COMMAND ${PRECISION_CODEGEN} ${CMAKE_CURRENT_SOURCE_DIR}/${_file}
        MAIN_DEPENDENCY ${_file}
      ) 
      list(APPEND generated_files_${_prec} "${file_tro}")
    endforeach()
  endif()
  list(APPEND generated_files ${generated_files_${_prec}})
endforeach()

message(STATUS "generate precisions: ${generated_files}")

add_library(dague_matrix STATIC 
    matrix.c 
    two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.c 
    tabular_distribution/tabular_distribution.c 
    sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.c
    ${generated_files})


