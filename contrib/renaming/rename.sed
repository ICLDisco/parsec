# data types
s/enum matrix_type/parsec_matrix_type_t/g
s/matrix_RealDouble/PARSEC_MATRIX_DOUBLE/g
s/matrix_RealFloat/PARSEC_MATRIX_FLOAT/g
s/matrix_Integer/PARSEC_MATRIX_INTEGER/g
s/matrix_Byte/PARSEC_MATRIX_BYTE/g
s/matrix_ComplexFloat/PARSEC_MATRIX_COMPLEX_FLOAT/g
s/matrix_ComplexDouble/PARSEC_MATRIX_COMPLEX_DOUBLE/g

# storage
s/enum matrix_storage/parsec_matrix_storage_t/g
s/matrix_Lapack/PARSEC_MATRIX_LAPACK/g
s/matrix_Tile/PARSEC_MATRIX_TILE/g

# matrix shapes
s/enum matrix_uplo/parsec_matrix_uplo_t/g
s/matrix_UpperLower/PARSEC_MATRIX_FULL/g
s/matrix_Upper/PARSEC_MATRIX_UPPER/g
s/matrix_Lower/PARSEC_MATRIX_LOWER/g

# vector shapes
s/matrix_VectorRow/PARSEC_VECTOR_DISTRIB_ROW/g
s/matrix_VectorCol/PARSEC_VECTOR_DISTRIB_COL/g
s/matrix_VectorDiag/PARSEC_VECTOR_DISTRIB_DIAG/g

# matrix implementations
s/parsec_tiled_matrix_dc/parsec_tiled_matrix/g
s/sym_two_dim_block_cyclic/parsec_matrix_sym_block_cyclic/g
s/two_dim_block_cyclic/parsec_matrix_block_cyclic/g
s/two_dim_tabular_/parsec_matrix_tabular_/g
s/two_dim_td_table_clone_table_structure/parsec_matrix_tabular_clone_table_structure/g

# symbols in matrix.h
s/tiled_matrix_submatrix/parsec_tiled_matrix_submatrix/g
s/parsec_matrix_create_data/parsec_tiled_matrix_create_data/g
s/parsec_matrix_add2arena/parsec_add2arena/g
s/parsec_matrix_del2arena/parsec_del2arena/g
s/parsec_matrix_data_/parsec_tiled_matrix_data_/g

# vector
s/vector_distrib_t/parsec_vector_two_dim_cyclic_distrib_t/g
s/vector_two_dim_cyclic_/parsec_vector_two_dim_cyclic_/g

# 2D grid
s/grid_2Dcyclic_/parsec_grid_2Dcyclic_/g

# matrix ops
s/tiled_matrix_unary_op_t/parsec_tiled_matrix_unary_op_t/g
s/tiled_matrix_binary_op_t/parsec_tiled_matrix_binary_op_t/g
