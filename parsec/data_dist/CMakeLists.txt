set(sources
  hash_datadist.c)

add_library(parsec_distribution ${sources})
if (MPI_C_FOUND)
    set_target_properties(parsec_distribution PROPERTIES COMPILE_FLAGS
                          "${MPI_C_COMPILE_FLAGS}")
endif(MPI_C_FOUND)

target_link_libraries(parsec_distribution parsec m)

install(TARGETS parsec_distribution
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib )
install(FILES
  hash_datadist.h
  DESTINATION include/parsec/data_dist )

add_subdirectory(matrix)

