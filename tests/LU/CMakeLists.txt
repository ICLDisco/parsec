if( PLASMA_FOUND )
  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src )
  
  add_executable(dgesv dgesv.c LU.c data_management.c)
  set_target_properties(dgesv PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dgesv PROPERTIES COMPILE_FLAGS "-DADD_ -DDPLASMA_EXECUTE")
  set_target_properties(dgesv PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(dgesv dplasma ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

  if( DPLASMA_MPI AND MPI_FOUND )
    add_executable(mpi_dgesv dgesv.c data_management.c LU.c)
    set_target_properties(mpi_dgesv PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_dgesv PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_dgesv PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_dgesv dplasma-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  endif( DPLASMA_MPI AND MPI_FOUND )

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT LU.c
  COMMAND ../../tools/dpc LU.jdf LU.c
  MAIN_DEPENDENCY LU.jdf
  DEPENDS dpc
  )
