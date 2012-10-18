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
int  dplasma_zgemm( dague_context_t *dague, const int transA, const int transB,
                    const dague_complex64_t alpha, const tiled_matrix_desc_t* A, const tiled_matrix_desc_t* B,
                    const dague_complex64_t beta,  tiled_matrix_desc_t* C);
int  dplasma_zhemm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo,
                    const dague_complex64_t alpha, const tiled_matrix_desc_t* A, const tiled_matrix_desc_t* B,
                    const double beta,  tiled_matrix_desc_t* C);
int  dplasma_zherk( dague_context_t *dague, const PLASMA_enum uplo, const PLASMA_enum trans,
                    const double alpha, const tiled_matrix_desc_t* A,
                    const double beta,  tiled_matrix_desc_t* C);
void dplasma_ztrmm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
                    const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

int dplasma_ztrdsm( dague_context_t *dague, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
int dplasma_ztrsm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
                   const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
void dplasma_ztrsmpl(    dague_context_t *dague, const tiled_matrix_desc_t *A,
                         const tiled_matrix_desc_t *L, const tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B);
void dplasma_ztrsmpl_sd( dague_context_t *dague, const tiled_matrix_desc_t *A,
                         const tiled_matrix_desc_t *L, tiled_matrix_desc_t *B);
int  dplasma_zsymm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo,
                    const dague_complex64_t alpha, const tiled_matrix_desc_t* A, const tiled_matrix_desc_t* B,
                    const double beta,  tiled_matrix_desc_t* C);

/* Lapack */
int    dplasma_zpotrf( dague_context_t *dague, const PLASMA_enum uplo, tiled_matrix_desc_t* A);
int    dplasma_zpotrs( dague_context_t *dague, const PLASMA_enum uplo, tiled_matrix_desc_t* A, tiled_matrix_desc_t* B);
int    dplasma_zposv ( dague_context_t *dague, const PLASMA_enum uplo, tiled_matrix_desc_t* A, tiled_matrix_desc_t* B);
int    dplasma_zhetrs( dague_context_t *dague, int uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, PLASMA_Complex64_t *U_but_vec, int level);
int    dplasma_zgetrf( dague_context_t *dague, tiled_matrix_desc_t* A, tiled_matrix_desc_t *IPIV );
int    dplasma_zgetrf_incpiv( dague_context_t *dague, tiled_matrix_desc_t* A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV );
int    dplasma_zgetrs( dague_context_t *dague, const PLASMA_enum trans, tiled_matrix_desc_t* A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zgetrs_incpiv( dague_context_t *dague, const PLASMA_enum trans, tiled_matrix_desc_t* A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zgesv ( dague_context_t *dague, tiled_matrix_desc_t* A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zgesv_incpiv ( dague_context_t *dague, tiled_matrix_desc_t* A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zgeqrf( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
int    dplasma_zgeqrf_param( dague_context_t *dague, qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT);
int    dplasma_zgelqf( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
int    dplasma_zungqr( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *Q);
int    dplasma_zungqr_param( dague_context_t *dague, qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *Q);
int    dplasma_zunmqr( dague_context_t *dague, PLASMA_enum side, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B);

int    dplasma_zgeadd( dague_context_t *dague, PLASMA_enum uplo, dague_complex64_t alpha, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
int    dplasma_zlacpy( dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
double dplasma_zlange( dague_context_t *dague, PLASMA_enum ntype, tiled_matrix_desc_t *A);
double dplasma_zlanhe( dague_context_t *dague, PLASMA_enum ntype, PLASMA_enum uplo, tiled_matrix_desc_t *A);
double dplasma_zlansy( dague_context_t *dague, PLASMA_enum ntype, PLASMA_enum uplo, tiled_matrix_desc_t *A);
int    dplasma_zlaset( dague_context_t *dague, PLASMA_enum uplo, dague_complex64_t alpha, dague_complex64_t beta, tiled_matrix_desc_t *A);
int    dplasma_zlaswp( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, int inc);
int    dplasma_zplghe( dague_context_t *dague, double            bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
int    dplasma_zplgsy( dague_context_t *dague, dague_complex64_t bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
int    dplasma_zplrnt( dague_context_t *dague,                                           tiled_matrix_desc_t *A, unsigned long long int seed);


/* Auxiliary routines */
int  dplasma_zprint( dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A);

int dplasma_zhebut( dague_context_t *dague, tiled_matrix_desc_t *A, PLASMA_Complex64_t **U_but_ptr, int level);
int dplasma_zhetrf(dague_context_t *dague, tiled_matrix_desc_t *A);

/***********************************************************
 *             Non-Blocking interface
 */
/* Level 3 Blas */
dague_object_t* dplasma_zgemm_New( const int transa, const int transb,
                                   const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                                   const PLASMA_Complex64_t beta,  tiled_matrix_desc_t *C);
dague_object_t* dplasma_zhemm_New( const PLASMA_enum side, const PLASMA_enum uplo,
                                   const dague_complex64_t alpha, const tiled_matrix_desc_t* A, const tiled_matrix_desc_t* B,
                                   const double beta,  tiled_matrix_desc_t* C);
dague_object_t* dplasma_zherk_New( const PLASMA_enum uplo, const PLASMA_enum trans,
                                   const double alpha, const tiled_matrix_desc_t* A,
                                   const double beta,  tiled_matrix_desc_t* C);
dague_object_t* dplasma_ztrmm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
                                   const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

dague_object_t* dplasma_ztrdsm_New( const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B );
dague_object_t* dplasma_ztrsm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
                                   const PLASMA_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
dague_object_t* dplasma_ztrsmpl_New(const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *L,
                                    const tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B);
dague_object_t* dplasma_zsymm_New( const PLASMA_enum side, const PLASMA_enum uplo,
                                   const dague_complex64_t alpha, const tiled_matrix_desc_t* A, const tiled_matrix_desc_t* B,
                                   const double beta,  tiled_matrix_desc_t* C);

/* Lapack */
dague_object_t* dplasma_zgetrf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, int* INFO);
dague_object_t* dplasma_zgetrf_incpiv_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, int* INFO);
dague_object_t* dplasma_zgeqrf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
dague_object_t* dplasma_zgeqrf_param_New(qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT);
dague_object_t* dplasma_zgelqf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
dague_object_t *dplasma_zherbt_New( PLASMA_enum uplo, int ib, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
dague_object_t* dplasma_zhbrdt_New(tiled_matrix_desc_t* A);
dague_object_t* dplasma_zpotrf_New( PLASMA_enum uplo, tiled_matrix_desc_t* A, int* INFO);
dague_object_t* dplasma_zungqr_New( tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *Q);
dague_object_t* dplasma_zungqr_param_New(qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *Q);
dague_object_t* dplasma_zunmqr_New( PLASMA_enum side, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B);

/* Lapack variants */
dague_object_t* dplasma_zgetrf_incpiv_sd_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *LIPIV, int* INFO);
dague_object_t* dplasma_zpotrf_rl_New(const PLASMA_enum uplo, tiled_matrix_desc_t* A, int* INFO);
dague_object_t* dplasma_zpotrf_ll_New(const PLASMA_enum uplo, tiled_matrix_desc_t* A, int* INFO);
dague_object_t* dplasma_zpotrfl_New(const PLASMA_enum looking, PLASMA_enum uplo, tiled_matrix_desc_t* A, int* INFO);
dague_object_t* dplasma_ztrsmpl_sd_New(const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *LIPIV, tiled_matrix_desc_t *B);

/* Auxiliary routines */
dague_object_t* dplasma_zgeadd_New( PLASMA_enum uplo, dague_complex64_t alpha, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
dague_object_t* dplasma_zlacpy_New( PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
dague_object_t* dplasma_zlaset_New( PLASMA_enum uplo, dague_complex64_t alpha, dague_complex64_t beta, tiled_matrix_desc_t *A);
dague_object_t* dplasma_zlange_New( PLASMA_enum ntype, int P, int Q, tiled_matrix_desc_t *A, double *norm);
dague_object_t* dplasma_zlaswp_New( tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, int inc);
dague_object_t* dplasma_zplrnt_New(                                           tiled_matrix_desc_t *A, unsigned long long int seed);
dague_object_t* dplasma_zplghe_New( double            bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
dague_object_t* dplasma_zplgsy_New( dague_complex64_t bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);


/*
 * Under development 
 */
/* Low-level nonblocking butterfly interface */
dague_object_t* dplasma_zhebut_New( tiled_matrix_desc_t *A, PLASMA_Complex64_t *U_but_vec, int it, int jt, int level, int *info);
dague_object_t* dplasma_zgebut_New( tiled_matrix_desc_t *A, PLASMA_Complex64_t *U_but_vec, int it, int jt, int level, int *info);
dague_object_t* dplasma_zgebmm_New( tiled_matrix_desc_t *A, PLASMA_Complex64_t *U_but_vec, int it, int jt, int level, int trans, int *info);


/* Low-level nonblocking LDL interface */
dague_object_t* dplasma_zhetrf_New( tiled_matrix_desc_t *A, int *info);
dague_object_t* dplasma_ztrmdm_New( tiled_matrix_desc_t *A);


/***********************************************************
 *               Destruct functions
 */
void dplasma_zgemm_Destruct( dague_object_t *o );
void dplasma_zhemm_Destruct( dague_object_t *o );
void dplasma_zherk_Destruct( dague_object_t *o );
void dplasma_ztrmm_Destruct( dague_object_t *o );
void dplasma_ztrdsm_Destruct( dague_object_t *o );
void dplasma_ztrsm_Destruct( dague_object_t *o );
void dplasma_ztrsmpl_Destruct( dague_object_t *o );
void dplasma_ztrsmpl_sd_Destruct( dague_object_t *o );
void dplasma_zsymm_Destruct( dague_object_t *o );

void dplasma_zgelqf_Destruct( dague_object_t *o );
void dplasma_zgeqrf_Destruct( dague_object_t *o );
void dplasma_zgeqrf_param_Destruct( dague_object_t *o );
void dplasma_zgetrf_Destruct( dague_object_t *o );
void dplasma_zgetrf_incpiv_Destruct( dague_object_t *o );
void dplasma_zgetrf_incpiv_sd_Destruct( dague_object_t *o );
void dplasma_zherbt_Destruct( dague_object_t *o );
void dplasma_zhbrdt_Destruct( dague_object_t* o );
void dplasma_zpotrf_Destruct( dague_object_t *o );
void dplasma_zungqr_Destruct( dague_object_t *o );
void dplasma_zungqr_param_Destruct( dague_object_t *o );
void dplasma_zunmqr_Destruct( dague_object_t *o );

void dplasma_zgeadd_Destruct( dague_object_t *o );
void dplasma_zlacpy_Destruct( dague_object_t *o );
void dplasma_zlaset_Destruct( dague_object_t *o );
void dplasma_zlange_Destruct( dague_object_t *o );
void dplasma_zlaswp_Destruct( dague_object_t *o );
void dplasma_zplrnt_Destruct( dague_object_t *o );
void dplasma_zplghe_Destruct( dague_object_t *o );
void dplasma_zplgsy_Destruct( dague_object_t *o );


void dplasma_zhebut_Destruct( dague_object_t *o );
void dplasma_zgebut_Destruct( dague_object_t *o );
void dplasma_zgebmm_Destruct( dague_object_t *o );
void dplasma_zhetrf_Destruct( dague_object_t *o );
void dplasma_ztrmdm_Destruct( dague_object_t *o );

/**********************************************************
 * Work in progress - to be move just before merge
 * (to avoid conflict when default is merge in the branch)
 */

int dplasma_zgetrf_hincpiv( dague_context_t *dague, qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* INFO);
int dplasma_zgetrf_hpp( dague_context_t *dague, qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* INFO);
int dplasma_zgetrf_hpp2( dague_context_t *dague, qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, tiled_matrix_desc_t *LT2, int* INFO);
int dplasma_zgetrf_hpp_multithrd( dague_context_t *dague, qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* INFO);
int dplasma_zgetrf_qrf( dague_context_t *dague, qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* lu_tab, int* INFO);
void dplasma_ztrsmpl_hincpiv( dague_context_t *dague, qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* INFO);
int dplasma_ztrsmpl_hpp( dague_context_t *dague, qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* INFO);
int dplasma_ztrsmpl_hpp2( dague_context_t *dague, qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, tiled_matrix_desc_t *LT2, int* INFO);
int dplasma_ztrsmpl_qrf( dague_context_t *dague, qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, tiled_matrix_desc_t *B, int* lu_tab);

dague_object_t* dplasma_zgetrf_hincpiv_New(qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* INFO);
dague_object_t* dplasma_zgetrf_hpp_New(qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* INFO);
dague_object_t* dplasma_zgetrf_hpp2_New(qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, tiled_matrix_desc_t *LT2, int* INFO);
dague_object_t* dplasma_zgetrf_hpp_multithrd_New(qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* INFO);
dague_object_t* dplasma_zgetrf_qrf_New(qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* lu_tab, int* INFO);

dague_object_t* dplasma_ztrsmpl_hincpiv_New( qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* INFO);
dague_object_t* dplasma_ztrsmpl_hpp_New(qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, int* INFO);
dague_object_t* dplasma_ztrsmpl_hpp2_New(qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, tiled_matrix_desc_t *LT2, int* INFO);
dague_object_t* dplasma_ztrsmpl_qrf_New(qr_piv_t *qrpiv, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *LT, tiled_matrix_desc_t *B, int* lu_tab);

void dplasma_zgetrf_hincpiv_Destruct( dague_object_t *o );
void dplasma_zgetrf_hpp_Destruct( dague_object_t *o );
void dplasma_zgetrf_hpp2_Destruct( dague_object_t *o );
void dplasma_zgetrf_hpp_multithrd_Destruct( dague_object_t *o );
void dplasma_zgetrf_qrf_Destruct( dague_object_t *o );

void dplasma_ztrsmpl_hincpiv_Destruct( dague_object_t *o );
void dplasma_ztrsmpl_hpp_Destruct( dague_object_t *o );
void dplasma_ztrsmpl_hpp2_Destruct( dague_object_t *o );
void dplasma_ztrsmpl_qrf_Destruct( dague_object_t *o );

#endif /* _DPLASMA_Z_H_ */
