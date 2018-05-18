set(sources
  hash_datadist.c)

add_library(parsec_data_dist OBJECT ${sources})
set_target_properties(parsec_data_dist PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (MPI_C_FOUND)
    set_target_properties(parsec_data_dist PROPERTIES COMPILE_FLAGS
                          "${MPI_C_COMPILE_FLAGS}")
    set_target_properties(parsec_data_dist PROPERTIES LINK_FLAGS
                          "${MPI_C_LINK_FLAGS}")
endif(MPI_C_FOUND)

add_subdirectory(matrix)

add_library(parsec_data INTERFACE)
target_sources(parsec_data INTERFACE $<TARGET_OBJECTS:parsec_data_dist>)
target_sources(parsec_data INTERFACE $<TARGET_OBJECTS:parsec_data_matrix>)

install(FILES
  hash_datadist.h
  DESTINATION include/parsec/data_dist )

