if( PLASMA_FOUND )
  
  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(pdgels dgels.c)
    set_target_properties(pdgels PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(pdgels PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(pdgels PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(pdgels dplasma-mpi dague-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  endif( DAGUE_MPI AND MPI_FOUND )

  add_executable(dgels dgels.c)
  set_target_properties(dgels PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dgels PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(dgels PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(dgels dplasma dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

endif( PLASMA_FOUND )

