#ifndef _DPLASMA_H_
#define _DPLASMA_H_

#include "data_dist/matrix/matrix.h"

void DAGUE_ztrmm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
		  const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

void DAGUE_ctrmm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
		  const PLASMA_Complex32_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

void DAGUE_dtrmm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
		  const double alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

void DAGUE_strmm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
		  const float alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);



dague_object_t* DAGUE_zgemm_New( const int transa, const int transb,
				 const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t* ddescA, const tiled_matrix_desc_t* ddescB,
				 const PLASMA_Complex64_t beta,  tiled_matrix_desc_t* ddescC);
dague_object_t* DAGUE_ztrmm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				 const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *work);
dague_object_t* DAGUE_ztrsm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				 const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
dague_object_t* DAGUE_zpotrf_rl_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_zpotrf_ll_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_zpotrf_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_zgetrf_sd_New(tiled_matrix_desc_t *LIPIV, tiled_matrix_desc_t* ddescA, int IB, int* INFO);
dague_object_t* DAGUE_zgetrf_New(tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t* ddescA, int IB, int* INFO);
dague_object_t* DAGUE_zgeqrf_New(tiled_matrix_desc_t *T, tiled_matrix_desc_t* ddescA, int IB);


dague_object_t* DAGUE_cgemm_New( const int transa, const int transb,
				 const PLASMA_Complex32_t alpha, const tiled_matrix_desc_t* ddescA, const tiled_matrix_desc_t* ddescB,
				 const PLASMA_Complex32_t beta,  tiled_matrix_desc_t* ddescC);
dague_object_t* DAGUE_ctrmm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				 const PLASMA_Complex32_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *work);
dague_object_t* DAGUE_ctrsm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				 const PLASMA_Complex32_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
dague_object_t* DAGUE_cpotrf_rl_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_cpotrf_ll_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_cpotrf_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_cgetrf_sd_New(tiled_matrix_desc_t *LIPIV, tiled_matrix_desc_t* ddescA, int IB, int* INFO);
dague_object_t* DAGUE_cgetrf_New(tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t* ddescA, int IB, int* INFO);
dague_object_t* DAGUE_cgeqrf_New(tiled_matrix_desc_t *T, tiled_matrix_desc_t* ddescA, int IB);



dague_object_t* DAGUE_dgemm_New( const int transa, const int transb,
				 const double alpha, const tiled_matrix_desc_t* ddescA, const tiled_matrix_desc_t* ddescB,
				 const double beta,  tiled_matrix_desc_t* ddescC);
dague_object_t* DAGUE_dtrmm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				 const double alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *work);
dague_object_t* DAGUE_dtrsm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				 const double alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
dague_object_t* DAGUE_dpotrf_rl_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_dpotrf_ll_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_dpotrf_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_dgetrf_sd_New(tiled_matrix_desc_t *LIPIV, tiled_matrix_desc_t* ddescA, int IB, int* INFO);
dague_object_t* DAGUE_dgetrf_New(tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t* ddescA, int IB, int* INFO);
dague_object_t* DAGUE_dgeqrf_New(tiled_matrix_desc_t *T, tiled_matrix_desc_t* ddescA, int IB);


dague_object_t* DAGUE_sgemm_New( const int transa, const int transb,
				 const float alpha, const tiled_matrix_desc_t* ddescA, const tiled_matrix_desc_t* ddescB,
				 const float beta,  tiled_matrix_desc_t* ddescC);
dague_object_t* DAGUE_strmm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				 const float alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *work);
dague_object_t* DAGUE_strsm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				 const float alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
dague_object_t* DAGUE_spotrf_rl_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_spotrf_ll_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_spotrf_New(char uplo, const tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* DAGUE_sgetrf_sd_New(tiled_matrix_desc_t *LIPIV, tiled_matrix_desc_t* ddescA, int IB, int* INFO);
dague_object_t* DAGUE_sgetrf_New(tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t* ddescA, int IB, int* INFO);
dague_object_t* DAGUE_sgeqrf_New(tiled_matrix_desc_t *T, tiled_matrix_desc_t* ddescA, int IB);

#endif
