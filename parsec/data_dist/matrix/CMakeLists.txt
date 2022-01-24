
#
# Sources generated from JDF
#
include(ParsecCompilePTG)

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

set(sources
    ${CMAKE_CURRENT_LIST_DIR}/matrix.c
    ${CMAKE_CURRENT_LIST_DIR}/matrixtypes.c
    ${CMAKE_CURRENT_LIST_DIR}/map_operator.c
    ${CMAKE_CURRENT_LIST_DIR}/two_dim_tabular.c
    ${CMAKE_CURRENT_LIST_DIR}/grid_2Dcyclic.c
    ${CMAKE_CURRENT_LIST_DIR}/two_dim_rectangle_cyclic.c
    ${CMAKE_CURRENT_LIST_DIR}/two_dim_rectangle_cyclic_band.c
    ${CMAKE_CURRENT_LIST_DIR}/sym_two_dim_rectangle_cyclic.c
    ${CMAKE_CURRENT_LIST_DIR}/sym_two_dim_rectangle_cyclic_band.c
    ${CMAKE_CURRENT_LIST_DIR}/vector_two_dim_cyclic.c
    ${CMAKE_CURRENT_LIST_DIR}/reduce_wrapper.c
    ${CMAKE_CURRENT_LIST_DIR}/apply_wrapper.c
    ${CMAKE_CURRENT_LIST_DIR}/subtile.c
)

set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf"
                    "${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf"
                    "${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf"
             APPEND PROPERTY PTGPP_COMPILE_OPTIONS "--Wremoteref")

target_ptg_sources(parsec PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf;${CMAKE_CURRENT_SOURCE_DIR}/diag_band_to_rect.jdf;${CMAKE_CURRENT_SOURCE_DIR}/apply.jdf")

target_sources(parsec PRIVATE ${sources})

add_subdirectory(redistribute)

install(FILES
  matrix.h
  two_dim_rectangle_cyclic.h
  two_dim_rectangle_cyclic_band.h
  sym_two_dim_rectangle_cyclic.h
  sym_two_dim_rectangle_cyclic_band.h
  vector_two_dim_cyclic.h
  two_dim_tabular.h
  grid_2Dcyclic.h
  subtile.h
  ${CMAKE_CURRENT_BINARY_DIR}/diag_band_to_rect.h
  DESTINATION ${PARSEC_INSTALL_INCLUDEDIR}/parsec/data_dist/matrix )


install(FILES
  deprecated/grid_2Dcyclic.h
  deprecated/matrix.h
  deprecated/two_dim_rectangle_cyclic.h
  deprecated/two_dim_rectangle_cyclic_band.h
  deprecated/sym_two_dim_rectangle_cyclic.h
  deprecated/sym_two_dim_rectangle_cyclic_band.h
  deprecated/vector_two_dim_cyclic.h
  deprecated/two_dim_tabular.h
  deprecated/grid_2Dcyclic.h
  DESTINATION ${PARSEC_INSTALL_INCLUDEDIR}/parsec/data_dist/matrix/deprecated )
