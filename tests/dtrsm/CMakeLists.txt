if( PLASMA_FOUND )

  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src)
  
  set( DTRSM_SRC
    trsm.c 
    dtrsm.c 
    trsm_LLN.c
    trsm_LLT.c
    trsm_LUN.c
    trsm_LUT.c
    trsm_RLN.c
    trsm_RLT.c
    trsm_RUN.c
    trsm_RUT.c
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

add_custom_command (
  OUTPUT trsm_LLT.c trsm_LLT.h
  COMMAND ../../tools/dague-compiler/dc -i trsm_LLT.jdf -o trsm_LLT
  MAIN_DEPENDENCY trsm_LLT.jdf
  DEPENDS dc
  )

add_custom_command (
  OUTPUT trsm_LUN.c trsm_LUN.h
  COMMAND ../../tools/dague-compiler/dc -i trsm_LUN.jdf -o trsm_LUN
  MAIN_DEPENDENCY trsm_LUN.jdf
  DEPENDS dc
  )

add_custom_command (
  OUTPUT trsm_LUT.c trsm_LUT.h
  COMMAND ../../tools/dague-compiler/dc -i trsm_LUT.jdf -o trsm_LUT
  MAIN_DEPENDENCY trsm_LUT.jdf
  DEPENDS dc
  )


add_custom_command (
  OUTPUT trsm_RLN.c trsm_RLN.h
  COMMAND ../../tools/dague-compiler/dc -i trsm_RLN.jdf -o trsm_RLN
  MAIN_DEPENDENCY trsm_RLN.jdf
  DEPENDS dc
  )


add_custom_command (
  OUTPUT trsm_RLT.c trsm_RLT.h
  COMMAND ../../tools/dague-compiler/dc -i trsm_RLT.jdf -o trsm_RLT
  MAIN_DEPENDENCY trsm_RLT.jdf
  DEPENDS dc
  )

add_custom_command (
  OUTPUT trsm_RUN.c trsm_RUN.h
  COMMAND ../../tools/dague-compiler/dc -i trsm_RUN.jdf -o trsm_RUN
  MAIN_DEPENDENCY trsm_RUN.jdf
  DEPENDS dc
  )

add_custom_command (
  OUTPUT trsm_RUT.c trsm_RUT.h
  COMMAND ../../tools/dague-compiler/dc -i trsm_RUT.jdf -o trsm_RUT
  MAIN_DEPENDENCY trsm_RUT.jdf
  DEPENDS dc
  )


