if( DPLASMA_MPI AND MPI_FOUND )
  add_executable(mpi_syn main.c syn.c)
  set_target_properties(mpi_syn PROPERTIES LINKER_LANGUAGE C)
  set_target_properties(mpi_syn PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_")
  set_target_properties(mpi_syn PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_C_LINK_FLAGS}")
  target_link_libraries(mpi_syn dplasma-mpi ${MPI_LIBRARIES} ${EXTRA_LIBS} -lm)
endif( DPLASMA_MPI AND MPI_FOUND )

add_executable(syn main.c syn.c)
set_target_properties(syn PROPERTIES LINKER_LANGUAGE C)
set_target_properties(syn PROPERTIES COMPILE_FLAGS "-DADD_")
set_target_properties(syn PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS}")
target_link_libraries(syn dplasma ${EXTRA_LIBS} -lm)

# add the command to generate the source code
add_custom_command (
  OUTPUT syn.c
  COMMAND ../../tools/dague-compiler/daguepp syn.jdf syn.c
  MAIN_DEPENDENCY syn.jdf
  DEPENDS daguepp
  )
 
