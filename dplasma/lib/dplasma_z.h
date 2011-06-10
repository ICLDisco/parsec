/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 */
#ifndef _DPLASMA_Z_H_
#define _DPLASMA_Z_H_

/***********************************************************
 *               Blocking interface 
 */
/* Level 3 Blas */
void dplasma_zgemm( dague_context_t *dague, const int transA, const int transB,
                    const Dague_Complex64_t alpha, const tiled_matrix_desc_t* ddescA, const tiled_matrix_desc_t* ddescB,
                    const Dague_Complex64_t beta,  tiled_matrix_desc_t* ddescC);
void dplasma_ztrmm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
		  const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
void dplasma_ztrsm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
		  const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
void dplasma_ztrsmpl( dague_context_t *dague, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *L, 
                      const tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B);

/* Lapack */
int  dplasma_zpotrf( dague_context_t *dague, const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA);
int  dplasma_zpotrs( dague_context_t *dague, const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA, tiled_matrix_desc_t* ddescB);
int  dplasma_zposv ( dague_context_t *dague, const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA, tiled_matrix_desc_t* ddescB);
int  dplasma_zgetrf( dague_context_t *dague, tiled_matrix_desc_t* A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV ); 
int  dplasma_zgetrs( dague_context_t *dague, const PLASMA_enum trans, tiled_matrix_desc_t* A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B ); 
int  dplasma_zgesv ( dague_context_t *dague, tiled_matrix_desc_t* A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B ); 
int  dplasma_zgeqrf( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T) ;
int  dplasma_zgelqf( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T) ;

int dplasma_zlaset( dague_context_t *dague, PLASMA_enum uplo, Dague_Complex64_t alpha, Dague_Complex64_t beta, tiled_matrix_desc_t *A); 
int dplasma_zplghe( dague_context_t *dague, Dague_Complex64_t bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
int dplasma_zplgsy( dague_context_t *dague, Dague_Complex64_t bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
int dplasma_zplrnt( dague_context_t *dague, tiled_matrix_desc_t *A, unsigned long long int seed);

/***********************************************************
 *             Non-Blocking interface
 */
/* Level 3 Blas */
dague_object_t* dplasma_zgemm_New( const int transa, const int transb,
				   const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
				   const PLASMA_Complex64_t beta,  tiled_matrix_desc_t *C);
dague_object_t* dplasma_ztrmm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				   const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *work);
dague_object_t* dplasma_ztrsm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
				   const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
dague_object_t* dplasma_ztrsmpl_New(const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *L,
                                    const tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B);

/* Lapack */
dague_object_t* dplasma_zgetrf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, int* INFO);
dague_object_t* dplasma_zgeqrf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
dague_object_t* dplasma_zgelqf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
dague_object_t *dplasma_zherbt_New( PLASMA_enum uplo, int ib, PLASMA_desc descA, tiled_matrix_desc_t *A, PLASMA_desc descT, tiled_matrix_desc_t *T);
dague_object_t* dplasma_zhbrdt_New(tiled_matrix_desc_t* A);
dague_object_t* dplasma_zpotrf_New(const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA, int* INFO);

/* Lapack variants */
dague_object_t* dplasma_zgetrf_sd_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *LIPIV, int* INFO);
dague_object_t* dplasma_zpotrf_rl_New(const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* dplasma_zpotrf_ll_New(const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA, int* INFO);
dague_object_t* dplasma_zpotrfl_New(const PLASMA_enum looking, const PLASMA_enum uplo, tiled_matrix_desc_t* ddescA, int* INFO);

/* Auxiliary routines */
dague_object_t* dplasma_zlaset_New( PLASMA_enum uplo, Dague_Complex64_t alpha, Dague_Complex64_t beta, tiled_matrix_desc_t *A);
dague_object_t* dplasma_zplrnt_New( tiled_matrix_desc_t *A, unsigned long long int seed);
dague_object_t* dplasma_zplghe_New( Dague_Complex64_t bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
dague_object_t* dplasma_zplgsy_New( Dague_Complex64_t bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);

/***********************************************************
 *               Destruct functions
 */
void dplasma_zgemm_Destruct( dague_object_t *o );
void dplasma_ztrmm_Destruct( dague_object_t *o );
void dplasma_ztrsm_Destruct( dague_object_t *o );
void dplasma_ztrsmpl_Destruct( dague_object_t *o );

void dplasma_zgelqf_Destruct( dague_object_t *o );
void dplasma_zgeqrf_Destruct( dague_object_t *o );
void dplasma_zherbt_Destruct( dague_object_t *o );
void dplasma_zhbrdt_Destruct( dague_object_t* o );
void dplasma_zpotrf_Destruct( dague_object_t *o );

void dplasma_zlaset_Destruct( dague_object_t *o );
void dplasma_zplrnt_Destruct( dague_object_t *o );
void dplasma_zplghe_Destruct( dague_object_t *o );
void dplasma_zplgsy_Destruct( dague_object_t *o );


#endif /* _DPLASMA_Z_H_ */
