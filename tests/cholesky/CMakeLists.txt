if( PLASMA_FOUND )

  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src)
  
  if( DPLASMA_MPI AND MPI_FOUND )
    add_executable(mpi_dposv_ll dist_exec.c data_management.c cholesky_ll.c)
    set_target_properties(mpi_dposv_ll PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_dposv_ll PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_dposv_ll PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_dposv_ll dplasma-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

	add_executable(mpi_dposv_rl dist_exec.c data_management.c cholesky_rl.c)
	set_target_properties(mpi_dposv_rl PROPERTIES LINKER_LANGUAGE Fortran)
	set_target_properties(mpi_dposv_rl PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
	set_target_properties(mpi_dposv_rl PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
	target_link_libraries(mpi_dposv_rl dplasma-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  endif( DPLASMA_MPI AND MPI_FOUND )

  add_executable(dposv_ll dist_exec.c data_management.c cholesky_ll.c)
  set_target_properties(dposv_ll PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dposv_ll PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(dposv_ll PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(dposv_ll dplasma ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

  add_executable(dposv_rl dist_exec.c data_management.c cholesky_rl.c)
  set_target_properties(dposv_rl PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dposv_rl PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(dposv_rl PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(dposv_rl dplasma ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT cholesky_rl.c
  COMMAND ../../tools/dpc cholesky_rl.jdf cholesky_rl.c
  MAIN_DEPENDENCY cholesky_rl.jdf
  DEPENDS dpc
  )
 
add_custom_command (
  OUTPUT cholesky_ll.c
  COMMAND ../../tools/dpc cholesky_ll.jdf cholesky_ll.c
  MAIN_DEPENDENCY cholesky_ll.jdf
  DEPENDS dpc
  )
 
