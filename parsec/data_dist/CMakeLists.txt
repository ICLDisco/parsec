set(sources
  hash_datadist.c)

if(NOT TARGET parsec_data_dist)
  add_library(parsec_data_dist OBJECT ${sources})
endif(NOT TARGET parsec_data_dist)

if(BUILD_SHARED_LIBS)
  set_target_properties(parsec_data_dist PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(BUILD_SHARED_LIBS)

if (MPI_C_FOUND)
  target_link_libraries(parsec_data_dist PRIVATE MPI::MPI_C)
endif(MPI_C_FOUND)

add_subdirectory(matrix)

add_library(parsec_data INTERFACE)
target_sources(parsec_data INTERFACE $<TARGET_OBJECTS:parsec_data_dist>)
target_sources(parsec_data INTERFACE $<TARGET_OBJECTS:parsec_data_matrix>)

install(FILES
  hash_datadist.h
  DESTINATION include/parsec/data_dist )

