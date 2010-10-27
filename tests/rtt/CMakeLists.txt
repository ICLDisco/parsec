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

add_executable(rtt rtt.c main.c)
set_target_properties(rtt PROPERTIES LINKER_LANGUAGE C)
set_target_properties(rtt COMPILE_FLAGS "-DADD_")
set_target_properties(rtt PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS}")
target_link_libraries(rtt dague)

if( DAGUE_MPI AND MPI_FOUND )
   add_executable(mpi_rtt rtt.c main.c)
   set_target_properties(mpi_rtt PROPERTIES LINKER_LANGUAGE C)
   set_target_properties(mpi_rtt PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI")
   set_target_properties(mpi_rtt PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_C_LINK_FLAGS}")
   target_link_libraries(mpi_rtt dague-mpi ${MPI_LIBRARIES})
endif( DAGUE_MPI AND MPI_FOUND )

add_custom_command (
  OUTPUT rtt.c rtt.h
  COMMAND ../../tools/dague-compiler/daguepp -i rtt.jdf -o rtt
  MAIN_DEPENDENCY rtt.jdf
  DEPENDS daguepp
  )
 
