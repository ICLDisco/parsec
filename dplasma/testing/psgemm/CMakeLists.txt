if( PLASMA_FOUND )

  include_directories( ../../lib ${PLASMA_INCLUDE_DIR} ${PLASMA_SRC_DIR} )
  
  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(psgemm testing_psgemm.c)
    set_target_properties(psgemm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(psgemm PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(psgemm PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(psgemm dplasma-mpi dague-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  endif( DAGUE_MPI AND MPI_FOUND )

  add_executable(sgemm testing_psgemm.c)
  set_target_properties(sgemm PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(sgemm PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(sgemm PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(sgemm dplasma dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

endif( PLASMA_FOUND )

