if( PLASMA_FOUND )
  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src )
  
  add_executable(dgels dgels.c QR.c)
  set_target_properties(dgels PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dgels PROPERTIES COMPILE_FLAGS "-I${PLASMA_SRC_DIR} -DADD_")
  set_target_properties(dgels PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(dgels dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(mpi_dgels dgels.c QR.c)
    set_target_properties(mpi_dgels PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_dgels PROPERTIES COMPILE_FLAGS "-I${PLASMA_SRC_DIR} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_dgels PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_dgels dague-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  endif( DAGUE_MPI AND MPI_FOUND )

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT QR.c QR.h
  COMMAND ../../tools/dague-compiler/dc -i QR.jdf -o QR
  MAIN_DEPENDENCY QR.jdf
  DEPENDS dc
  )
