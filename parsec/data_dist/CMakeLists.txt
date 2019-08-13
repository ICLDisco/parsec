set(sources
  hash_datadist.c)

add_subdirectory(matrix)

add_library(parsec-data STATIC ${sources})
target_sources(parsec-data PUBLIC $<TARGET_OBJECTS:parsec-data-matrix>)
set_target_properties(parsec-data PROPERTIES POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})
target_include_directories(parsec-data PRIVATE ${MPI_C_INCLUDE_PATH})

if (MPI_C_FOUND)
  target_link_libraries(parsec-data INTERFACE MPI::MPI_C)
  target_include_directories(parsec-data INTERFACE ${MPI_C_INCLUDE_PATH})
endif(MPI_C_FOUND)

install(FILES
  hash_datadist.h
  DESTINATION include/parsec/data_dist )

