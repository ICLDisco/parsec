
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
    ${CMAKE_CURRENT_LIST_DIR}/subtile.c
)

if( TARGET parsec-ptgpp )
  list(APPEND sources
       ${CMAKE_CURRENT_LIST_DIR}/reduce_wrapper.c
       ${CMAKE_CURRENT_LIST_DIR}/apply_wrapper.c)
  set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf"
                      "${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf"
                      "${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf"
               APPEND PROPERTY PTGPP_COMPILE_OPTIONS "--Wremoteref")

  target_ptg_sources(parsec PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf;${CMAKE_CURRENT_SOURCE_DIR}/diag_band_to_rect.jdf;${CMAKE_CURRENT_SOURCE_DIR}/apply.jdf")
  set_property(TARGET parsec
               APPEND PROPERTY
                      PRIVATE_HEADER_H data_dist/matrix/diag_band_to_rect.h)
endif( TARGET parsec-ptgpp )

target_sources(parsec PRIVATE ${sources})

add_subdirectory(redistribute)

set_property(TARGET parsec
             APPEND PROPERTY
                    PRIVATE_HEADER_H data_dist/matrix/matrix.h
                                     data_dist/matrix/two_dim_rectangle_cyclic.h
                                     data_dist/matrix/two_dim_rectangle_cyclic_band.h
                                     data_dist/matrix/sym_two_dim_rectangle_cyclic.h
                                     data_dist/matrix/sym_two_dim_rectangle_cyclic_band.h
                                     data_dist/matrix/vector_two_dim_cyclic.h
                                     data_dist/matrix/two_dim_tabular.h
                                     data_dist/matrix/grid_2Dcyclic.h
                                     data_dist/matrix/subtile.h)

# Install the deprecated headers.
set_property(TARGET parsec
             APPEND PROPERTY
                    PRIVATE_HEADER_H data_dist/matrix/deprecated/grid_2Dcyclic.h
                                     data_dist/matrix/deprecated/sym_two_dim_rectangle_cyclic_band.h
                                     data_dist/matrix/deprecated/two_dim_rectangle_cyclic_band.h
                                     data_dist/matrix/deprecated/two_dim_tabular.h
                                     data_dist/matrix/deprecated/matrix.h
                                     data_dist/matrix/deprecated/sym_two_dim_rectangle_cyclic.h
                                     data_dist/matrix/deprecated/two_dim_rectangle_cyclic.h
                                     data_dist/matrix/deprecated/vector_two_dim_cyclic.h)
