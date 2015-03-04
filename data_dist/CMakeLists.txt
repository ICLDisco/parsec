set(sources
  hash_datadist.c)

add_library(dague_distribution ${sources})
if (MPI_C_FOUND)
    set_target_properties(dague_distribution PROPERTIES COMPILE_FLAGS
                          "${MPI_C_COMPILE_FLAGS}")
endif(MPI_C_FOUND)
install(TARGETS dague_distribution
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib )

add_subdirectory(matrix)

