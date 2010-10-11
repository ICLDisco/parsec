#ifndef _DPLASMA_H_
#define _DPLASMA_H_

#include "data_dist/matrix/matrix.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

dague_object_t* DAGUE_zgemm_New( int transa, int transb, int m, int n, int k,
				 PLASMA_Complex64_t alpha, const tiled_matrix_desc_t* ddescA,
				 const tiled_matrix_desc_t* ddescB,
				 PLASMA_Complex64_t beta, tiled_matrix_desc_t* ddescC);
dague_object_t* DAGUE_ztrmm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
				 PLASMA_Complex64_t alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B);
dague_object_t* DAGUE_ztrsm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
				 PLASMA_Complex64_t alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B);

dague_object_t* DAGUE_cgemm_New( int transa, int transb, int m, int n, int k,
				 PLASMA_Complex32_t alpha, const tiled_matrix_desc_t* ddescA,
				 const tiled_matrix_desc_t* ddescB,
				 PLASMA_Complex32_t beta, tiled_matrix_desc_t* ddescC);
dague_object_t* DAGUE_ctrmm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
				 PLASMA_Complex32_t alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B);
dague_object_t* DAGUE_ctrsm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
				 PLASMA_Complex32_t alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B);

dague_object_t* DAGUE_dgemm_New( int transa, int transb, int m, int n, int k,
				 double alpha, const tiled_matrix_desc_t* ddescA,
				 const tiled_matrix_desc_t* ddescB,
				 double beta, tiled_matrix_desc_t* ddescC);
dague_object_t* DAGUE_dtrmm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
				 double alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B);
dague_object_t* DAGUE_dtrsm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
				 double alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B);

dague_object_t* DAGUE_sgemm_New( int transa, int transb, int m, int n, int k,
				 float alpha, const tiled_matrix_desc_t* ddescA,
				 const tiled_matrix_desc_t* ddescB,
				 float beta, tiled_matrix_desc_t* ddescC);
dague_object_t* DAGUE_strmm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
				 float alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B);
dague_object_t* DAGUE_strsm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
				 float alpha, two_dim_block_cyclic_t *A, two_dim_block_cyclic_t *B);

#endif
