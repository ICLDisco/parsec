if( PLASMA_FOUND )
  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src )
  
  add_executable(testing_tsqr testing_tsqr.c TSQR.c)
  set_target_properties(testing_tsqr PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(testing_tsqr PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(testing_tsqr PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(testing_tsqr dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(mpi_testing_tsqr testing_tsqr.c TSQR.c)
    set_target_properties(mpi_testing_tsqr PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_testing_tsqr PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_testing_tsqr PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_testing_tsqr dague-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  endif( DAGUE_MPI AND MPI_FOUND )

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT TSQR.c TSQR.h
  COMMAND ../../tools/dague-compiler/dc -i TSQR.jdf -o TSQR
  MAIN_DEPENDENCY TSQR.jdf
  DEPENDS dc
  )
