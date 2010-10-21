if( PLASMA_FOUND )
  

  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(mpi_dgesv dgesv.c)
    set_target_properties(mpi_dgesv PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_dgesv PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_dgesv PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_dgesv dplasma-mpi dague-mpi dague_distribution_matrix-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

    add_executable(mpi_dgesv_sd dgesv_sd.c)
    set_target_properties(mpi_dgesv_sd PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_dgesv_sd PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_dgesv_sd PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_dgesv_sd dplasma-mpi dague-mpi dague_distribution_matrix-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

  else( DAGUE_MPI AND MPI_FOUND )
    add_executable(dgesv dgesv.c)
    set_target_properties(dgesv PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(dgesv PROPERTIES COMPILE_FLAGS "-DADD_")
    set_target_properties(dgesv PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(dgesv dplasma dague dague_distribution_matrix ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

    add_executable(dgesv_sd dgesv_sd.c)
    set_target_properties(dgesv_sd PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(dgesv_sd PROPERTIES COMPILE_FLAGS "-DADD_")
    set_target_properties(dgesv_sd PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(dgesv_sd dplasma dague dague_distribution_matrix ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
  endif( DAGUE_MPI AND MPI_FOUND )

endif( PLASMA_FOUND )

