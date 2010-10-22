/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 */
#ifndef _DPLASMA_Z_H_
#define _DPLASMA_Z_H_

/*
 * Blocking interface 
 */
/* Level 3 Blas */
void dplasma_zgemm( dague_context_t *dague, const int transA, const int transB,
		    const Dague_Complex64_t alpha, const tiled_matrix_desc_t* ddescA, const tiled_matrix_desc_t* ddescB,
		    const Dague_Complex64_t beta,  tiled_matrix_desc_t* ddescC);
void dplasma_ztrmm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
		  const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
void dplasma_ztrsm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
		  const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

/* Lapack */
int  dplasma_zpotrf( dague_context_t *dague, const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA);
int  dplasma_zgetrf( dague_context_t *dague, tiled_matrix_desc_t* ddescA, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV ); 
int  dplasma_zgeqrf( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T) ;

/*
 * Non-Blocking interface
 */
/* Level 3 Blas */
dague_object_t* dplasma_zgemm_New( const int transa, const int transb,
				   const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
				   const PLASMA_Complex64_t beta,  tiled_matrix_desc_t *C);
dague_object_t* dplasma_ztrmm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				   const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *work);
dague_object_t* dplasma_ztrsm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				   const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

/* Lapack */
dague_object_t* dplasma_zpotrf_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* dplasma_zgetrf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, int* INFO);
dague_object_t* dplasma_zgeqrf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);

/* Lapack variants */
dague_object_t* dplasma_zpotrf_rl_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* dplasma_zpotrf_ll_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* dplasma_zgetrf_sd_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *LIPIV, int* INFO);

#endif /* _DPLASMA_Z_H_ */
