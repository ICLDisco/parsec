if( PLASMA_FOUND )

  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(pdgemm testing_pdgemm.c)
    set_target_properties(pdgemm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(pdgemm PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(pdgemm PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(pdgemm dplasma-mpi dague-mpi dague_distribution_matrix-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  else( DAGUE_MPI AND MPI_FOUND )
    add_executable(dgemm testing_pdgemm.c)
    set_target_properties(dgemm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(dgemm PROPERTIES COMPILE_FLAGS "-DADD_")
    set_target_properties(dgemm PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(dgemm dplasma dague dague_distribution_matrix ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
  endif( DAGUE_MPI AND MPI_FOUND )

endif( PLASMA_FOUND )

