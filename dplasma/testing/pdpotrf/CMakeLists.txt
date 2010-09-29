if( PLASMA_FOUND )

  include_directories( ../../lib ${PLASMA_INCLUDE_DIR})
  
  if( DAGUE_MPI AND MPI_FOUND )

    add_executable(mpi_llt llt.c)
    set_target_properties(mpi_llt PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_llt PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_llt PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_llt dplasma-mpi dague-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

    # add_executable(mpi_testio testio.c)
    # set_target_properties(mpi_testio PROPERTIES LINKER_LANGUAGE Fortran)
    # set_target_properties(mpi_testio PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    # set_target_properties(mpi_testio PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    # target_link_libraries(mpi_testio dague-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

  endif( DAGUE_MPI AND MPI_FOUND )

    add_executable(llt llt.c)
    set_target_properties(llt PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(llt PROPERTIES COMPILE_FLAGS "-DADD_")
   set_target_properties(llt PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(llt dplasma dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
 
endif( PLASMA_FOUND )

