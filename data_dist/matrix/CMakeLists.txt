
#
# Sources depending on Precision 
#
include(RulesPrecisions)
include(RulesJDF)
set(generated_files "")
set(generated_jdf "")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

jdf_rules(generated_jdf "${CMAKE_CURRENT_SOURCE_DIR}/reduce_col.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce_row.jdf;${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf;${CMAKE_CURRENT_SOURCE_DIR}/diag_band_to_rect.jdf")

set( precision_files
  zmatrix.c
  ztwoDBC.c
)
precisions_rules(generated_files
                 "${precision_files}"
                 PRECISIONS "+;s;d;c;z"
                 TARGETDIR "generated")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)

if (MPI_FOUND)
    add_library(dague_distribution_matrix-mpi STATIC
        matrix.c
        map_operator.c
        tabular_distribution.c
        grid_2Dcyclic.c
        two_dim_rectangle_cyclic.c
        sym_two_dim_rectangle_cyclic.c
        scalapack_convert.c reduce_wrapper.c
        ${generated_files} ${generated_jdf})
    set_target_properties(dague_distribution_matrix-mpi PROPERTIES COMPILE_FLAGS
                            "${MPI_COMPILE_FLAGS} -DUSE_MPI")
    install(TARGETS dague_distribution_matrix-mpi ARCHIVE DESTINATION lib)
else()
    add_library(dague_distribution_matrix STATIC 
        matrix.c 
        map_operator.c
        precision.c 
        tabular_distribution.c 
        grid_2Dcyclic.c
        two_dim_rectangle_cyclic.c 
        sym_two_dim_rectangle_cyclic.c
        reduce_wrapper.c
        ${generated_files} ${generated_jdf})
    install(TARGETS dague_distribution_matrix ARCHIVE DESTINATION lib)
endif()

