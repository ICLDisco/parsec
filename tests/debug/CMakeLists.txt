if( PLASMA_FOUND )

  include_directories( ${PLASMA_INCLUDE_DIR} ${PLASMA_INCLUDE_DIR}/../src)
  
  if( DPLASMA_MPI AND MPI_FOUND )
    add_executable(mpi_multiple main.c multiple.c)
    set_target_properties(mpi_multiple PROPERTIES LINKER_LANGUAGE C)
    set_target_properties(mpi_multiple PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_multiple PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_C_LINK_FLAGS}")
    target_link_libraries(mpi_multiple dplasma-mpi ${MPI_LIBRARIES} ${EXTRA_LIBS} -lm)
 endif( DPLASMA_MPI AND MPI_FOUND )

  if( DPLASMA_MPI AND MPI_FOUND )
    add_executable(mpi_types main2.c types.c)
    set_target_properties(mpi_types PROPERTIES LINKER_LANGUAGE C)
    set_target_properties(mpi_types PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_types PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_C_LINK_FLAGS}")
    target_link_libraries(mpi_types dplasma-mpi ${MPI_LIBRARIES} ${EXTRA_LIBS} -lm)
 endif( DPLASMA_MPI AND MPI_FOUND )

  if( DPLASMA_MPI AND MPI_FOUND )
    add_executable(mpi_pingpong main_pingpong.c pingpong.c)
    set_target_properties(mpi_pingpong PROPERTIES LINKER_LANGUAGE C)
    set_target_properties(mpi_pingpong PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
    set_target_properties(mpi_pingpong PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_C_LINK_FLAGS}")
    target_link_libraries(mpi_pingpong dplasma-mpi ${MPI_LIBRARIES} ${EXTRA_LIBS} -lm)
 endif( DPLASMA_MPI AND MPI_FOUND )

endif( PLASMA_FOUND )

# add the command to generate the source code
add_custom_command (
  OUTPUT multiple.c
  COMMAND ../../tools/dpc multiple.jdf multiple.c
  MAIN_DEPENDENCY multiple.jdf
  DEPENDS dpc
  )
 
add_custom_command (
  OUTPUT types.c
  COMMAND ../../tools/dpc types.jdf types.c
  MAIN_DEPENDENCY types.jdf
  DEPENDS dpc
  )
  
add_custom_command (
  OUTPUT pingpong.c
  COMMAND ../../tools/dpc pingpong.jdf pingpong.c
  MAIN_DEPENDENCY pingpong.jdf
  DEPENDS dpc
  )
 
