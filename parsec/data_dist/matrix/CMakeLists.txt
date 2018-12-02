
#
# Sources generated from JDF
#
include(RulesJDF)
set(generated_jdf "")


SET_PROPERTY(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf PROPERTY ADDITIONAL_PARSECPP_CFLAGS "--Wremoteref")
SET_PROPERTY(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf PROPERTY ADDITIONAL_PARSECPP_CFLAGS "--Wremoteref")
SET_PROPERTY(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf PROPERTY ADDITIONAL_PARSECPP_CFLAGS "--Wremoteref")
jdf_rules(generated_jdf "${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf;${CMAKE_CURRENT_SOURCE_DIR}/diag_band_to_rect.jdf;${CMAKE_CURRENT_SOURCE_DIR}/apply.jdf")

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

if(NOT PARSEC_COMPILE_INPLACE)
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
endif(NOT PARSEC_COMPILE_INPLACE)

set(sources
        matrix.c
        matrixtypes.c
        map_operator.c
        two_dim_tabular.c
        grid_2Dcyclic.c
        two_dim_rectangle_cyclic.c
        two_dim_rectangle_cyclic_band.c
        sym_two_dim_rectangle_cyclic.c
        sym_two_dim_rectangle_cyclic_band.c
        vector_two_dim_cyclic.c
        reduce_wrapper.c
        apply_wrapper.c
        subtile.c
        ${generated_jdf}
)

if (MPI_C_FOUND)
    LIST(APPEND sources scalapack_convert.c)
endif(MPI_C_FOUND)

add_library(parsec_data_matrix OBJECT ${sources})
if(BUILD_SHARED_LIBS)
  set_target_properties(parsec_data_matrix PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif(BUILD_SHARED_LIBS)
if (MPI_C_FOUND)
    set_target_properties(parsec_data_matrix PROPERTIES COMPILE_FLAGS
                          "${MPI_C_COMPILE_FLAGS}")
    set_target_properties(parsec_data_matrix PROPERTIES LINK_FLAGS
                          "${MPI_C_LINK_FLAGS}")
endif(MPI_C_FOUND)

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
  DESTINATION include/parsec/data_dist/matrix )
