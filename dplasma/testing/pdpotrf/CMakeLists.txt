if( PLASMA_FOUND )

  set(DPOTRF_REQUIRED_LIBRARIES ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
  set(GPU_SUPPORT "")

 
  if( DAGUE_MPI AND MPI_FOUND )

    add_executable(mpi_dposv dposv.c)
    set_target_properties(mpi_dposv PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_dposv PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_dposv PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_dposv dplasma-mpi dague-mpi dague_distribution_matrix-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

    #add_executable(mpi_tab_dposv tab_dposv.c)
    #set_target_properties(mpi_tab_dposv PROPERTIES LINKER_LANGUAGE Fortran)
    #set_target_properties(mpi_tab_dposv PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    #set_target_properties(mpi_tab_dposv PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    #target_link_libraries(mpi_tab_dposv dplasma-mpi dague-mpi dague_distribution_matrix-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

    # add_executable(mpi_testio testio.c)
    # set_target_properties(mpi_testio PROPERTIES LINKER_LANGUAGE Fortran)
    # set_target_properties(mpi_testio PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    # set_target_properties(mpi_testio PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    # target_link_libraries(mpi_testio dague-mpi dague_distribution_matrix-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

  endif( DAGUE_MPI AND MPI_FOUND )

    add_executable(dposv dposv.c)
    set_target_properties(dposv PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(dposv PROPERTIES COMPILE_FLAGS "-DADD_")
   set_target_properties(dposv PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(dposv dplasma dague dague_distribution_matrix ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
 
endif( PLASMA_FOUND )

