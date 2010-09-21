if( PLASMA_FOUND )

  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_SRC_DIR} )
  
  if( DAGUE_MPI AND MPI_FOUND )
    add_executable(psgemm psgemm.c sgemm.c)
    set_target_properties(psgemm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(psgemm PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(psgemm PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(psgemm dague-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  endif( DAGUE_MPI AND MPI_FOUND )

  add_executable(sgemm psgemm.c sgemm.c)
  set_target_properties(sgemm PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(sgemm PROPERTIES COMPILE_FLAGS "-DADD_")
  set_target_properties(sgemm PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(sgemm dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT sgemm.c sgemm.h
  COMMAND ../../tools/dague-compiler/dc -i sgemm.jdf -o sgemm
  MAIN_DEPENDENCY sgemm.jdf
  DEPENDS dc
  )

