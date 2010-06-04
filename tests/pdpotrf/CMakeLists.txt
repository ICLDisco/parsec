if( PLASMA_FOUND )

  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src)
  
  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(mpi_dposv_ll dposv.c data_management.c cholesky_ll.c)
    set_target_properties(mpi_dposv_ll PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_dposv_ll PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_dposv_ll PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_dposv_ll dague-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

	add_executable(mpi_dposv_rl dposv.c data_management.c cholesky_rl.c)
	set_target_properties(mpi_dposv_rl PROPERTIES LINKER_LANGUAGE Fortran)
	set_target_properties(mpi_dposv_rl PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
	set_target_properties(mpi_dposv_rl PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
	target_link_libraries(mpi_dposv_rl dague-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  endif( DAGUE_MPI AND MPI_FOUND )

  add_executable(dposv_ll dposv.c data_management.c cholesky_ll.c)
  set_target_properties(dposv_ll PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dposv_ll PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(dposv_ll PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(dposv_ll dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

  add_executable(dposv_rl dposv.c data_management.c cholesky_rl.c)
  set_target_properties(dposv_rl PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dposv_rl PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(dposv_rl PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(dposv_rl dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT cholesky_rl.c
  COMMAND ../../tools/dague-compiler/dc -i cholesky_rl.jdf -o cholesky_rl
  MAIN_DEPENDENCY cholesky_rl.jdf
  DEPENDS dc
  )
 
add_custom_command (
  OUTPUT cholesky_ll.c
  COMMAND ../../tools/dague-compiler/dc -i cholesky_ll.jdf -o cholesky_ll
  MAIN_DEPENDENCY cholesky_ll.jdf
  DEPENDS dc
  )
 
