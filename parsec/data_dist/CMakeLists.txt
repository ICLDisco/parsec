set(sources
  hash_datadist.c)

add_library(parsec_data_dist OBJECT ${sources})
if (MPI_C_FOUND)
    set_target_properties(parsec_data_dist PROPERTIES COMPILE_FLAGS
                          "${MPI_C_COMPILE_FLAGS}")
endif(MPI_C_FOUND)

add_subdirectory(matrix)
add_library(parsec_data STATIC $<TARGET_OBJECTS:parsec_data_matrix>
                               $<TARGET_OBJECTS:parsec_data_dist>)

install(FILES
  hash_datadist.h
  DESTINATION include/parsec/data_dist )

