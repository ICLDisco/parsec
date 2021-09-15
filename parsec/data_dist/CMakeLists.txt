set(sources
  ${CMAKE_CURRENT_LIST_DIR}/hash_datadist.c)

add_subdirectory(matrix)

target_sources(parsec PRIVATE ${sources})

set_property(TARGET parsec
             APPEND PROPERTY
             PRIVATE_HEADER_H data_dist/hash_datadist.h)

