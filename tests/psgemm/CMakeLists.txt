if( PLASMA_FOUND )

  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src)
  
  if( DPLASMA_MPI AND MPI_FOUND )
    add_executable(psgemm psgemm.c data_management.c gemm.c)
    set_target_properties(psgemm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(psgemm PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -I${PLASMA_SRC_DIR} -DADD_ -DUSE_MPI")
    set_target_properties(psgemm PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(psgemm dplasma-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  endif( DPLASMA_MPI AND MPI_FOUND )

  add_executable(sgemm psgemm.c data_management.c gemm.c)
  set_target_properties(sgemm PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(sgemm PROPERTIES COMPILE_FLAGS "-I${PLASMA_SRC_DIR} -DADD_")
  set_target_properties(sgemm PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(sgemm dplasma ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT gemm.c
  COMMAND ../../tools/dpc gemm.jdf gemm.c
  MAIN_DEPENDENCY gemm.jdf
  DEPENDS dpc
  )
