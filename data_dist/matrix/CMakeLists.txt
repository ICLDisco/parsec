
#
# Sources depending on Precision 
#
include(RulesPrecisions)
set(generated_files "")
include_directories(".")

set( precision_files
  zmatrix.c
  two_dim_rectangle_cyclic/ztwoDBC.c
)
precisions_rules(generated_files "+;s;d" "${precision_files}")
#message(STATUS "${generated_files}")

if (DAGUE_MPI AND MPI_FOUND)
    add_library(dague_distribution_matrix-mpi STATIC
        matrix.c
        tabular_distribution/tabular_distribution.c
        two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.c
        sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.c
	scalapack_convert.c
        ${generated_files})
    set_target_properties(dague_distribution_matrix-mpi PROPERTIES COMPILE_FLAGS
                            "${MPI_COMPILE_FLAGS} -DUSE_MPI")
    install(TARGETS dague_distribution_matrix-mpi ARCHIVE DESTINATION lib)
else()
    add_library(dague_distribution_matrix STATIC 
        matrix.c 
        precision.c 
        tabular_distribution/tabular_distribution.c 
        two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.c 
        sym_two_dim_rectangle_cyclic/sym_two_dim_rectangle_cyclic.c
        ${generated_files})
    install(TARGETS dague_distribution_matrix ARCHIVE DESTINATION lib)
endif()

