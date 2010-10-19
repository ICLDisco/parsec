if( PLASMA_FOUND )

  if( DAGUE_MPI AND MPI_FOUND )

    add_executable(mpi_trsm trsm.c )
    set_target_properties(mpi_trsm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_trsm PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_trsm PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_trsm dplasma-mpi dague-mpi dague_distribution_matrix-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

  endif( DAGUE_MPI AND MPI_FOUND )

    add_executable(trsm trsm.c )
    set_target_properties(trsm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(trsm PROPERTIES COMPILE_FLAGS "-DADD_")
    set_target_properties(trsm PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(trsm dplasma dague dague_distribution_matrix ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
 
endif( PLASMA_FOUND )

