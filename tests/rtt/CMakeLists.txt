if( MPI_FOUND )
   add_executable(mpi_rtt rtt.c rtt_wrapper.c rtt_data.c main.c)
   set_target_properties(mpi_rtt PROPERTIES LINKER_LANGUAGE C)
   set_target_properties(mpi_rtt PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS} ${LOCAL_C_LINK_FLAGS} ${HWLOC_LIB}")
  target_link_libraries(mpi_rtt dague-mpi ${MPI_LIBRARIES})
else(MPI_FOUND)
  add_executable(rtt rtt.c rtt_wrapper.c rtt_data.c main.c)
  set_target_properties(rtt PROPERTIES LINKER_LANGUAGE C)
  set_target_properties(rtt PROPERTIES LINK_FLAGS "${LOCAL_C_LINK_FLAGS} ${HWLOC_LIB}")
  target_link_libraries(rtt dague)
endif( MPI_FOUND )

add_custom_command (
  OUTPUT rtt.c rtt.h
  COMMAND ../../tools/dague-compiler/daguepp -i rtt.jdf -o rtt
  MAIN_DEPENDENCY rtt.jdf
  DEPENDS daguepp
  )
 
