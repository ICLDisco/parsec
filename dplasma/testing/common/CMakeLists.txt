set(SOURCES 
  testscommon.c 
  )

if (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasmatest-mpi
    ${SOURCES}
    )
  set_target_properties(dplasmatest-mpi 
    PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI"
    )
  install(TARGETS 
    dplasmatest-mpi ARCHIVE DESTINATION lib
    )
else (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasmatest 
    ${SOURCES}
    )
  set_target_properties(dplasmatest 
    PROPERTIES COMPILE_FLAGS "-DADD_")
  install(TARGETS 
    dplasmatest ARCHIVE DESTINATION lib)
endif (DAGUE_MPI AND MPI_FOUND)

