if( PLASMA_FOUND )

  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src)
  
  set( DTRSM_SRC
    trsm.c 
    dtrsm.c 
    trsm_LLN.c
    # trsm_LLT.c
    # trsm_LUN.c
    # trsm_LUT.c
    # trsm_RLN.c
    # trsm_RLT.c
    # trsm_RUN.c
    # trsm_RUT.c
    )
      
  if( DAGUE_MPI AND MPI_FOUND )

    add_executable(mpi_trsm 
      ${DTRSM_SRC} )
    set_target_properties(mpi_trsm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(mpi_trsm PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_trsm PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(mpi_trsm dague-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})

  endif( DAGUE_MPI AND MPI_FOUND )

    add_executable(trsm 
      ${DTRSM_SRC} )
    set_target_properties(trsm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(trsm PROPERTIES COMPILE_FLAGS "-DADD_")
    set_target_properties(trsm PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(trsm dague ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})
 
endif( PLASMA_FOUND )

# add the command to generate the source code

add_custom_command (
  OUTPUT trsm_LLN.c trsm_LLN.h
  COMMAND ../../tools/dague-compiler/dc -i trsm_LLN.jdf -o trsm_LLN
  MAIN_DEPENDENCY trsm_LLN.jdf
  DEPENDS dc
  )
