
#
# Sources depending on Precision 
#
include(RulesPrecisions)
include(RulesJDF)
set(generated_files "")
set(generated_jdf "")
include_directories(".")

jdf_rules(generated_jdf "${CMAKE_CURRENT_SOURCE_DIR}/reduce.jdf")

set( precision_files
  zmatrix.c
  two_dim_rectangle_cyclic/ztwoDBC.c
)
precisions_rules(generated_files "+;s;d;c;z" "${precision_files}")

if (MPI_FOUND)
    add_library(dague_distribution_matrix-mpi STATIC
        matrix.c
        apply_all.c
        tabular_distribution/tabular_distribution.c
        grid_2Dcyclic.c
        two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.c
        sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.c
        scalapack_convert.c
        ${generated_files} ${generated_jdf})
    set_target_properties(dague_distribution_matrix-mpi PROPERTIES COMPILE_FLAGS
                            "${MPI_COMPILE_FLAGS} -DUSE_MPI")
    install(TARGETS dague_distribution_matrix-mpi ARCHIVE DESTINATION lib)
else()
    add_library(dague_distribution_matrix STATIC 
        matrix.c 
        apply_all.c
        precision.c 
        tabular_distribution/tabular_distribution.c 
        grid_2Dcyclic.c
        two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.c 
        sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.c
        ${generated_files} ${generated_jdf})
    install(TARGETS dague_distribution_matrix ARCHIVE DESTINATION lib)
endif()

