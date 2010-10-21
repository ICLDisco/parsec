set(JDF
  gemm_NN.jdf gemm_TN.jdf gemm_NT.jdf gemm_TT.jdf
  trsm_LLN.jdf trsm_LLT.jdf trsm_LUN.jdf trsm_LUT.jdf trsm_RLN.jdf trsm_RLT.jdf trsm_RUN.jdf trsm_RUT.jdf
  trmm_LLN.jdf trmm_LLT.jdf trmm_LUN.jdf trmm_LUT.jdf trmm_RLN.jdf trmm_RLT.jdf trmm_RUN.jdf trmm_RUT.jdf
  potrf_rl.jdf potrf_ll.jdf getrf.jdf getrf_sd.jdf geqrf.jdf
  )

set(SOURCES
  trsm_wrapper.c gemm_wrapper.c trmm_wrapper.c potrf_wrapper.c getrf_wrapper.c geqrf_wrapper.c
)

set( OLDSRC 
  TSQR.jdf 
  sgeqrt.jdf
)
set(EXTRA_SOURCES memory_pool.c dplasmaaux.c)

set(DAGUEPP ../../tools/dague-compiler/daguepp)
set(generated_files)

# JDF generation 
set(PRECISIONS d s)
foreach(_prec ${PRECISIONS})
  set(generated_files_${_prec})
  foreach(_file ${JDF})
    string(REPLACE ".jdf" "" file_tro ${_file})
    add_custom_command(
      OUTPUT ${_prec}${file_tro}.c ${_prec}${file_tro}.h
      COMMAND ${DAGUEPP} -i ${_file} -o ${_prec}${file_tro}
      MAIN_DEPENDENCY ${_file}
      DEPENDS daguepp
      ) 
    list(APPEND generated_files_${_prec} "${_prec}${file_tro}.c")
  endforeach()
endforeach()

#  precision generation 
foreach(_prec ${PRECISIONS})
  foreach(_file ${SOURCES})
    add_custom_command(
      OUTPUT ${_prec}${_file}
      COMMAND cp ${_file} ${_prec}${_file}
      MAIN_DEPENDENCY ${_file}
      ) 
    list(APPEND generated_files_${_prec} "${_prec}${_file}")
  endforeach()
endforeach()

foreach(_prec ${PRECISIONS})
  list(APPEND generated_files ${generated_files_${_prec}})
endforeach()

foreach(_file ${OLDSRC})
  string(REPLACE ".jdf" "" file_tro ${_file})
  add_custom_command(
    OUTPUT ${file_tro}.c ${file_tro}.h
    COMMAND ${DAGUEPP} -i ${_file} -o ${file_tro}
    MAIN_DEPENDENCY ${_file}
    DEPENDS daguepp
    )
  list(APPEND generated_files "${file_tro}.c")
endforeach()

set_source_files_properties( ${generated_files_z}
  PROPERTIES COMPILE_FLAGS  "-DDAGCOMPLEX -DDAGDOUBLE"
)
set_source_files_properties( ${generated_files_c}
  PROPERTIES COMPILE_FLAGS  "-DDAGCOMPLEX -DDAGSINGLE"
)
set_source_files_properties( ${generated_files_d}
  PROPERTIES COMPILE_FLAGS  "-DDAGREAL -DDAGDOUBLE"
)
set_source_files_properties( ${generated_files_s}
  PROPERTIES COMPILE_FLAGS  "-DDAGREAL -DDAGSINGLE"
)

add_library(dplasma ${generated_files} ${EXTRA_SOURCES})
set_target_properties(dplasma PROPERTIES COMPILE_FLAGS "-DADD_")
install(TARGETS dplasma ARCHIVE DESTINATION lib)

if (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma-mpi ${generated_files} ${EXTRA_SOURCES})
  set_target_properties(dplasma-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
  install(TARGETS dplasma-mpi ARCHIVE DESTINATION lib)
endif (DAGUE_MPI AND MPI_FOUND)
