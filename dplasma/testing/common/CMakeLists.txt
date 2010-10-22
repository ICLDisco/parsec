set(SOURCES testscommon.c)

if (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma_testscommon-mpi
    ${SOURCES}
    )
  set_target_properties(dplasma_testscommon-mpi 
    PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DADD_ -DUSE_MPI"
    )
  install(TARGETS 
    dplasma_testscommon-mpi ARCHIVE DESTINATION lib
    )
else (DAGUE_MPI AND MPI_FOUND)
  add_library(dplasma_testscommon
    ${SOURCES}
    )
  set_target_properties(dplasma_testscommon 
    PROPERTIES COMPILE_FLAGS "-DADD_")
  install(TARGETS 
    dplasma_testscommon ARCHIVE DESTINATION lib)
endif (DAGUE_MPI AND MPI_FOUND)

