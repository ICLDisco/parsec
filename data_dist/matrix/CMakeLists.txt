
#
# Sources depending on Precision
#
include(RulesPrecisions)
include(RulesJDF)
set(generated_files "")
set(generated_jdf "")


SET_PROPERTY(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf PROPERTY ADDITIONAL_DAGUEPP_CFLAGS "--Wremoteref")
SET_PROPERTY(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf PROPERTY ADDITIONAL_DAGUEPP_CFLAGS "--Wremoteref")
SET_PROPERTY(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf PROPERTY ADDITIONAL_DAGUEPP_CFLAGS "--Wremoteref")
jdf_rules(generated_jdf "${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf;${CMAKE_CURRENT_SOURCE_DIR}/diag_band_to_rect.jdf")

set( precision_files
  ztwoDBC.c
)
precisions_rules_py(generated_files
                 "${precision_files}"
                 PRECISIONS "s;d;c;z"
)

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

if(NOT DAGUE_COMPILE_INPLACE)
  include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}")
endif(NOT DAGUE_COMPILE_INPLACE)

set(sources
        matrix.c
        map_operator.c
        two_dim_tabular.c
        grid_2Dcyclic.c
        two_dim_rectangle_cyclic.c
        sym_two_dim_rectangle_cyclic.c
        vector_two_dim_cyclic.c
        reduce_wrapper.c
        ${generated_files}
        ${generated_jdf}
)
if( NOT HAVE_COMPLEX_H )
  list(APPEND sources precision.c)
endif()

if (MPI_C_FOUND)
    LIST(APPEND $sources scalapack_convert.c)
endif(MPI_C_FOUND)

add_library(dague_distribution_matrix STATIC
            ${sources})
if (MPI_C_FOUND)
    set_target_properties(dague_distribution_matrix PROPERTIES COMPILE_FLAGS
                          "${MPI_C_COMPILE_FLAGS}")
endif(MPI_C_FOUND)
install(TARGETS dague_distribution_matrix ARCHIVE DESTINATION lib)

