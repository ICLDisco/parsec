
#
# Sources generated from JDF
#
include(ParsecCompilePTG)

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

if (MPI_C_FOUND)
  LIST(APPEND sources ${CMAKE_CURRENT_LIST_DIR}/scalapack_convert.c)
endif(MPI_C_FOUND)

if(NOT TARGET parsec-data-matrix)
  add_library(parsec-data-matrix OBJECT ${sources})
endif(NOT TARGET parsec-data-matrix)

set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf"
                    "${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf"
                    "${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf"
             APPEND PROPERTY PTGPP_COMPILE_OPTIONS "--Wremoteref")

target_ptg_sources(parsec-data-matrix PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf;${CMAKE_CURRENT_SOURCE_DIR}/diag_band_to_rect.jdf;${CMAKE_CURRENT_SOURCE_DIR}/apply.jdf")

set_target_properties(parsec-data-matrix PROPERTIES POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})
target_include_directories(parsec-data-matrix PUBLIC ${MPI_C_INCLUDE_PATH}) # ${CMAKE_CURRENT_BINARY_DIR}) added by target_ptg_sources

add_dependencies(parsec parsec-data-matrix)
target_sources(parsec PRIVATE $<TARGET_OBJECTS:parsec-data-matrix>)

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
  DESTINATION include/parsec/data_dist/matrix )
