
#
# Sources generated from JDF
#
include(RulesJDF)
set(generated_jdf "")


SET_PROPERTY(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf PROPERTY ADDITIONAL_PARSECPP_CFLAGS "--Wremoteref")
SET_PROPERTY(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf PROPERTY ADDITIONAL_PARSECPP_CFLAGS "--Wremoteref")
SET_PROPERTY(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf PROPERTY ADDITIONAL_PARSECPP_CFLAGS "--Wremoteref")
jdf_rules(generated_jdf "${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf;${CMAKE_CURRENT_SOURCE_DIR}/diag_band_to_rect.jdf")

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
        sym_two_dim_rectangle_cyclic.c
        vector_two_dim_cyclic.c
        reduce_wrapper.c
        subtile.c
        ${generated_jdf}
)

if (MPI_C_FOUND)
    LIST(APPEND sources scalapack_convert.c)
endif(MPI_C_FOUND)

add_library(parsec_data
            ${sources})
if (MPI_C_FOUND)
    set_target_properties(parsec_data PROPERTIES COMPILE_FLAGS
                          "${MPI_C_COMPILE_FLAGS}")
    set_target_properties(parsec_data PROPERTIES LINK_FLAGS
                          "${MPI_C_LINK_FLAGS}")
endif(MPI_C_FOUND)

target_link_libraries(parsec_data
  parsec_distribution
  ${MPI_C_LIBRARIES}
)

install(TARGETS parsec_data
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)

install(FILES
  matrix.h
  two_dim_rectangle_cyclic.h
  sym_two_dim_rectangle_cyclic.h
  vector_two_dim_cyclic.h
  two_dim_tabular.h
  grid_2Dcyclic.h
  subtile.h
  DESTINATION include/parsec/data_dist/matrix )
