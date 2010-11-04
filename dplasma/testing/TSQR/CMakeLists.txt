if( PLASMA_FOUND )
  
  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(mpi_testing_tsqr testing_tsqr.c)
    set_target_properties(mpi_testing_tsqr PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_testing_tsqr PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_")
    set_target_properties(mpi_testing_tsqr PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_testing_tsqr dplasma-mpi dague-mpi dague_distribution_matrix-mpi ${PLASMA_LDFLAGS} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  else( DAGUE_MPI AND MPI_FOUND )  
    add_executable(testing_tsqr testing_tsqr.c)
    set_target_properties(testing_tsqr PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(testing_tsqr PROPERTIES COMPILE_FLAGS "-DADD_")
    set_target_properties(testing_tsqr PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(testing_tsqr dplasma dague dague_distribution_matrix ${PLASMA_LDFLAGS} ${EXTRA_LIBS})
  endif( DAGUE_MPI AND MPI_FOUND )

endif( PLASMA_FOUND )

