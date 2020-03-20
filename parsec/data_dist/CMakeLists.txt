set(sources
  ${CMAKE_CURRENT_LIST_DIR}/hash_datadist.c)

add_subdirectory(matrix)

target_sources(parsec PRIVATE ${sources})

install(FILES
  hash_datadist.h
  DESTINATION ${PARSEC_INSTALL_INCLUDEDIR}/parsec/data_dist )

