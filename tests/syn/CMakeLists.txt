if( PLASMA_FOUND )

  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src)
  
  if( DPLASMA_MPI AND MPI_FOUND )
    add_executable(mpi_syn main.c data_management.c syn.c)
    set_target_properties(mpi_syn PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_syn PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_syn PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_syn dplasma-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
 endif( DPLASMA_MPI AND MPI_FOUND )

  add_executable(syn main.c data_management.c syn.c)
  set_target_properties(syn PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(syn PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(syn PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(syn dplasma ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT syn.c
  COMMAND ../../tools/dpc syn.jdf syn.c
  MAIN_DEPENDENCY syn.jdf
  DEPENDS dpc
  )
 
