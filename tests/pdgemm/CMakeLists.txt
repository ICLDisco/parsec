if( PLASMA_FOUND )

  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src)
  
  if( DAGUE_MPI AND MPI_FOUND )
    add_library(dgemmbox-mpi dgemm_NN.c dgemm_NT.c dgemm_TN.c dgemm_TT.c pdgemm.c )
    set_target_properties(dgemmbox-mpi PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")

    add_executable(pdgemm testing_pdgemm.c)
    set_target_properties(pdgemm PROPERTIES LINKER_LANGUAGE Fortran)
    set_target_properties(pdgemm PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -I${PLASMA_SRC_DIR} -DADD_ -DUSE_MPI")
    set_target_properties(pdgemm PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_FORTRAN_LINK_FLAGS}")
    target_link_libraries(pdgemm dague-mpi dgemmbox-mpi ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS})
  endif( DAGUE_MPI AND MPI_FOUND )

  add_library(dgemmbox dgemm_NN.c dgemm_NT.c dgemm_TN.c dgemm_TT.c pdgemm.c)
  set_target_properties(dgemmbox PROPERTIES COMPILE_FLAGS "-DADD_")

  add_executable(dgemm pdgemm.c testing_pdgemm.c)
  set_target_properties(dgemm PROPERTIES LINKER_LANGUAGE Fortran)
  set_target_properties(dgemm PROPERTIES COMPILE_FLAGS "-I${PLASMA_SRC_DIR} -DADD_")
  set_target_properties(dgemm PROPERTIES LINK_FLAGS "${LOCAL_FORTRAN_LINK_FLAGS}")
  target_link_libraries(dgemm dague dgemmbox ${PLASMA_LIBRARIES} ${BLAS_LIBRARIES} ${EXTRA_LIBS})

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT dgemm_NN.c dgemm_NN.h
  COMMAND ../../tools/dague-compiler/dc -i dgemm_NN.jdf -o dgemm_NN
  MAIN_DEPENDENCY dgemm_NN.jdf
  DEPENDS dc
  )
add_custom_command (
  OUTPUT dgemm_NT.c dgemm_NT.h
  COMMAND ../../tools/dague-compiler/dc -i dgemm_NT.jdf -o dgemm_NT
  MAIN_DEPENDENCY dgemm_NT.jdf
  DEPENDS dc
  )
add_custom_command (
  OUTPUT dgemm_TN.c dgemm_TN.h
  COMMAND ../../tools/dague-compiler/dc -i dgemm_TN.jdf -o dgemm_TN
  MAIN_DEPENDENCY dgemm_TN.jdf
  DEPENDS dc
  )
add_custom_command (
  OUTPUT dgemm_TT.c dgemm_TT.h
  COMMAND ../../tools/dague-compiler/dc -i dgemm_TT.jdf -o dgemm_TT
  MAIN_DEPENDENCY dgemm_TT.jdf
  DEPENDS dc
  )
