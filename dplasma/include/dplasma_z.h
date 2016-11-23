/*
 * Copyright (c) 2010-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 */
#ifndef _DPLASMA_Z_H_
#define _DPLASMA_Z_H_

#include "data_dist/matrix/matrix.h"

/***********************************************************
 *               Blocking interface
 */
/* Level 2 Blas */
int dplasma_zgeadd( parsec_context_t *parsec, PLASMA_enum trans, parsec_complex64_t alpha,
                    const tiled_matrix_desc_t *A, parsec_complex64_t beta, tiled_matrix_desc_t *B);
int dplasma_zgerc( parsec_context_t *parsec, parsec_complex64_t alpha,
                   const tiled_matrix_desc_t *X, const tiled_matrix_desc_t *Y, tiled_matrix_desc_t *A);
int dplasma_zgeru( parsec_context_t *parsec, parsec_complex64_t alpha,
                   const tiled_matrix_desc_t *X, const tiled_matrix_desc_t *Y, tiled_matrix_desc_t *A);
int dplasma_ztradd( parsec_context_t *parsec, PLASMA_enum uplo, PLASMA_enum trans, parsec_complex64_t alpha,
                    const tiled_matrix_desc_t *A, parsec_complex64_t beta, tiled_matrix_desc_t *B);

/* Level 3 Blas */
int dplasma_zgemm( parsec_context_t *parsec, PLASMA_enum transA, PLASMA_enum transB,
                   parsec_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                   parsec_complex64_t beta,  tiled_matrix_desc_t *C);
int dplasma_zhemm( parsec_context_t *parsec, PLASMA_enum side, PLASMA_enum uplo,
                   parsec_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                   parsec_complex64_t beta,  tiled_matrix_desc_t *C);
int dplasma_zherk( parsec_context_t *parsec, PLASMA_enum uplo, PLASMA_enum trans,
                   double alpha, const tiled_matrix_desc_t *A,
                   double beta,  tiled_matrix_desc_t *C);
int dplasma_zher2k( parsec_context_t *parsec, PLASMA_enum uplo, PLASMA_enum trans,
                    parsec_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                    double beta,  tiled_matrix_desc_t *C);
int dplasma_zsymm( parsec_context_t *parsec, PLASMA_enum side, PLASMA_enum uplo,
                   parsec_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                   parsec_complex64_t beta,  tiled_matrix_desc_t *C);
int dplasma_zsyrk( parsec_context_t *parsec, PLASMA_enum uplo, PLASMA_enum trans,
                   parsec_complex64_t alpha, const tiled_matrix_desc_t *A,
                   parsec_complex64_t beta,  tiled_matrix_desc_t *C);
int dplasma_zsyr2k( parsec_context_t *parsec, PLASMA_enum uplo, PLASMA_enum trans,
                    parsec_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                    parsec_complex64_t beta,  tiled_matrix_desc_t *C);
int dplasma_ztrmm( parsec_context_t *parsec, PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                   parsec_complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
int dplasma_ztrsm( parsec_context_t *parsec, PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                   parsec_complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

/* Level 3 Blas extensions */
int dplasma_ztrsmpl( parsec_context_t *parsec, const tiled_matrix_desc_t *A,
                     const tiled_matrix_desc_t *L, const tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B);

/* Lapack */
int    dplasma_zgelqf( parsec_context_t *parsec, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
int    dplasma_zgelqf_param( parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT);
int    dplasma_zgelqs( parsec_context_t *parsec, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B );
int    dplasma_zgelqs_param( parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B );
int    dplasma_zgels( parsec_context_t *parsec, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B );
int    dplasma_zgeqrf( parsec_context_t *parsec, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
int    dplasma_zgeqrf_param( parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT);
int    dplasma_zgeqrf_rec( parsec_context_t *parsec, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, int hnb);
int    dplasma_zgeqrs( parsec_context_t *parsec, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B );
int    dplasma_zgeqrs_param( parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B );
int    dplasma_zgesv ( parsec_context_t *parsec, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zgesv_incpiv ( parsec_context_t *parsec, tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zgetrf( parsec_context_t *parsec, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV );
int    dplasma_zgetrf_incpiv( parsec_context_t *parsec, tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV );
int    dplasma_zgetrf_nopiv(  parsec_context_t *parsec, tiled_matrix_desc_t *A );
int    dplasma_zgetrs( parsec_context_t *parsec, const PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zgetrs_incpiv( parsec_context_t *parsec, const PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zlauum( parsec_context_t *parsec, PLASMA_enum uplo, tiled_matrix_desc_t *A );
int    dplasma_zpoinv( parsec_context_t *parsec, PLASMA_enum uplo, tiled_matrix_desc_t *A);
int    dplasma_zpoinv_sync( parsec_context_t *parsec, PLASMA_enum uplo, tiled_matrix_desc_t *A);
int    dplasma_zposv ( parsec_context_t *parsec, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
int    dplasma_zpotrf( parsec_context_t *parsec, PLASMA_enum uplo, tiled_matrix_desc_t *A);
int    dplasma_zpotrf_rec( parsec_context_t *parsec, PLASMA_enum uplo, tiled_matrix_desc_t *A, int hmb );
int    dplasma_zpotrs( parsec_context_t *parsec, PLASMA_enum uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
int    dplasma_zpotri( parsec_context_t *parsec, PLASMA_enum uplo, tiled_matrix_desc_t *A);
int    dplasma_ztrtri( parsec_context_t *parsec, PLASMA_enum uplo, PLASMA_enum diag, tiled_matrix_desc_t *A );
int    dplasma_zunglq( parsec_context_t *parsec, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *Q);
int    dplasma_zunglq_param( parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *Q);
int    dplasma_zungqr( parsec_context_t *parsec, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *Q);
int    dplasma_zungqr_param( parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *Q);
int    dplasma_zunmlq( parsec_context_t *parsec, PLASMA_enum side, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B);
int    dplasma_zunmlq_param( parsec_context_t *parsec, PLASMA_enum side, PLASMA_enum trans, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B );
int    dplasma_zunmqr( parsec_context_t *parsec, PLASMA_enum side, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B);
int    dplasma_zunmqr_param( parsec_context_t *parsec, PLASMA_enum side, PLASMA_enum trans, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B );

/* Lapack Auxiliary */
int    dplasma_zlacpy( parsec_context_t *parsec, PLASMA_enum uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
double dplasma_zlange( parsec_context_t *parsec, PLASMA_enum ntype, const tiled_matrix_desc_t *A);
double dplasma_zlanhe( parsec_context_t *parsec, PLASMA_enum ntype, PLASMA_enum uplo, const tiled_matrix_desc_t *A);
double dplasma_zlanm2( parsec_context_t *parsec, const tiled_matrix_desc_t *A, int *info);
double dplasma_zlansy( parsec_context_t *parsec, PLASMA_enum ntype, PLASMA_enum uplo, const tiled_matrix_desc_t *A);
double dplasma_zlantr( parsec_context_t *parsec, PLASMA_enum ntype, PLASMA_enum uplo, PLASMA_enum diag, const tiled_matrix_desc_t *A);
int    dplasma_zlascal(parsec_context_t *parsec, PLASMA_enum type, parsec_complex64_t alpha, tiled_matrix_desc_t *A);
int    dplasma_zlaset( parsec_context_t *parsec, PLASMA_enum uplo, parsec_complex64_t alpha, parsec_complex64_t beta, tiled_matrix_desc_t *A);
int    dplasma_zlaswp( parsec_context_t *parsec, tiled_matrix_desc_t *A, const tiled_matrix_desc_t *IPIV, int inc);
int    dplasma_zlatms( parsec_context_t *parsec, PLASMA_enum mtxtype, double cond, tiled_matrix_desc_t *A, unsigned long long int seed);
int    dplasma_zplghe( parsec_context_t *parsec, double             bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
int    dplasma_zplgsy( parsec_context_t *parsec, parsec_complex64_t bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
int    dplasma_zplrnt( parsec_context_t *parsec, int diagdom,                               tiled_matrix_desc_t *A, unsigned long long int seed);

/* Auxiliary routines available only through synchronous interface */
int    dplasma_zpltmg( parsec_context_t *parsec, PLASMA_enum mtxtype, tiled_matrix_desc_t *A, unsigned long long int seed);
int    dplasma_zprint( parsec_context_t *parsec, PLASMA_enum uplo, const tiled_matrix_desc_t *A);

/***********************************************************
 *             Non-Blocking interface
 */
/* Level 2 Blas */
parsec_handle_t* dplasma_zgeadd_New( PLASMA_enum trans, parsec_complex64_t alpha, const tiled_matrix_desc_t *A, parsec_complex64_t beta, tiled_matrix_desc_t *B);
parsec_handle_t* dplasma_zgerc_New ( parsec_complex64_t alpha, const tiled_matrix_desc_t *X, const tiled_matrix_desc_t *Y, tiled_matrix_desc_t *A);
parsec_handle_t* dplasma_zgeru_New ( parsec_complex64_t alpha, const tiled_matrix_desc_t *X, const tiled_matrix_desc_t *Y, tiled_matrix_desc_t *A);
parsec_handle_t* dplasma_ztradd_New( PLASMA_enum uplo, PLASMA_enum trans, parsec_complex64_t alpha, const tiled_matrix_desc_t *A, parsec_complex64_t beta, tiled_matrix_desc_t *B);

/* Level 3 Blas */
parsec_handle_t* dplasma_zgemm_New( PLASMA_enum transa, PLASMA_enum transb,
                                   parsec_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                                   parsec_complex64_t beta,  tiled_matrix_desc_t *C);
parsec_handle_t* dplasma_zhemm_New( PLASMA_enum side, PLASMA_enum uplo,
                                   parsec_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                                   parsec_complex64_t beta,  tiled_matrix_desc_t *C);
parsec_handle_t* dplasma_zherk_New( PLASMA_enum uplo, PLASMA_enum trans,
                                   double alpha, const tiled_matrix_desc_t *A,
                                   double beta,  tiled_matrix_desc_t *C);
parsec_handle_t* dplasma_zher2k_New( PLASMA_enum uplo, PLASMA_enum trans,
                                    parsec_complex64_t alpha, const tiled_matrix_desc_t *A,
                                    const tiled_matrix_desc_t *B,
                                    double beta,  tiled_matrix_desc_t *C);
parsec_handle_t* dplasma_zsymm_New( PLASMA_enum side, PLASMA_enum uplo,
                                   parsec_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                                   parsec_complex64_t beta,  tiled_matrix_desc_t *C);
parsec_handle_t* dplasma_zsyrk_New( PLASMA_enum uplo, PLASMA_enum trans,
                                   parsec_complex64_t alpha, const tiled_matrix_desc_t *A,
                                   parsec_complex64_t beta,  tiled_matrix_desc_t *C);
parsec_handle_t* dplasma_zsyr2k_New( PLASMA_enum uplo, PLASMA_enum trans,
                                    parsec_complex64_t alpha, const tiled_matrix_desc_t *A,
                                    const tiled_matrix_desc_t *B,
                                    parsec_complex64_t beta,  tiled_matrix_desc_t *C);
parsec_handle_t* dplasma_ztrmm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                                   parsec_complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
parsec_handle_t* dplasma_ztrsm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                                   const parsec_complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

/* Level 3 Blas extensions */
parsec_handle_t* dplasma_ztrsmpl_New(const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *L,
                                    const tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B);

/* Lapack */
parsec_handle_t* dplasma_zgelqf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
parsec_handle_t* dplasma_zgelqf_param_New(dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT);
parsec_handle_t* dplasma_zgeqrf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
parsec_handle_t* dplasma_zgeqrf_param_New(dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT);
parsec_handle_t* dplasma_zgetrf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, int *INFO);
parsec_handle_t* dplasma_zgetrf_incpiv_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, int *INFO);
parsec_handle_t* dplasma_zgetrf_nopiv_New(tiled_matrix_desc_t *A, int *INFO);
parsec_handle_t* dplasma_zlauum_New( PLASMA_enum uplo, tiled_matrix_desc_t *A);
parsec_handle_t* dplasma_zpoinv_New( PLASMA_enum uplo, tiled_matrix_desc_t *A, int *INFO);
parsec_handle_t* dplasma_zpotrf_New( PLASMA_enum uplo, tiled_matrix_desc_t *A, int *INFO);
parsec_handle_t* dplasma_ztrtri_New( PLASMA_enum uplo, PLASMA_enum diag, tiled_matrix_desc_t *A, int *info);
parsec_handle_t* dplasma_zunglq_New( tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *Q);
parsec_handle_t* dplasma_zunglq_param_New(dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *Q);
parsec_handle_t* dplasma_zungqr_New( tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *Q);
parsec_handle_t* dplasma_zungqr_param_New(dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *Q);
parsec_handle_t* dplasma_zunmlq_New( PLASMA_enum side, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B);
parsec_handle_t* dplasma_zunmlq_param_New( PLASMA_enum side, PLASMA_enum trans, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B);
parsec_handle_t* dplasma_zunmqr_New( PLASMA_enum side, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B);
parsec_handle_t* dplasma_zunmqr_param_New( PLASMA_enum side, PLASMA_enum trans, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B);

/* Auxiliary routines */
parsec_handle_t* dplasma_zlacpy_New( PLASMA_enum uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
parsec_handle_t* dplasma_zlange_New( PLASMA_enum ntype, const tiled_matrix_desc_t *A, double *norm);
parsec_handle_t* dplasma_zlanhe_New( PLASMA_enum ntype, PLASMA_enum uplo, const tiled_matrix_desc_t *A, double *result);
parsec_handle_t* dplasma_zlanm2_New( const tiled_matrix_desc_t *A, double *norm, int *info );
parsec_handle_t* dplasma_zlansy_New( PLASMA_enum ntype, PLASMA_enum uplo, const tiled_matrix_desc_t *A, double *result);
parsec_handle_t* dplasma_zlantr_New( PLASMA_enum ntype, PLASMA_enum uplo, PLASMA_enum diag, const tiled_matrix_desc_t *A, double *result);
parsec_handle_t* dplasma_zlascal_New( PLASMA_enum type, parsec_complex64_t alpha, tiled_matrix_desc_t *A);
parsec_handle_t* dplasma_zlaset_New( PLASMA_enum uplo, parsec_complex64_t alpha, parsec_complex64_t beta, tiled_matrix_desc_t *A);
parsec_handle_t* dplasma_zlaswp_New( tiled_matrix_desc_t *A, const tiled_matrix_desc_t *IPIV, int inc);
parsec_handle_t* dplasma_zplghe_New( double            bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
parsec_handle_t* dplasma_zplgsy_New( parsec_complex64_t bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
parsec_handle_t* dplasma_zplrnt_New( int diagdom,                              tiled_matrix_desc_t *A, unsigned long long int seed);

/* Sub-kernels to recursive DAGs */
parsec_handle_t* dplasma_zgeqrfr_geqrt_New(tiled_matrix_desc_t *A,  tiled_matrix_desc_t *T,  void *work);
parsec_handle_t* dplasma_zgeqrfr_tsmqr_New(tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2, tiled_matrix_desc_t *V, tiled_matrix_desc_t *T, void *work);
parsec_handle_t* dplasma_zgeqrfr_tsqrt_New(tiled_matrix_desc_t *A1, tiled_matrix_desc_t *A2, tiled_matrix_desc_t *T, void *tau, void *work);
parsec_handle_t* dplasma_zgeqrfr_unmqr_New(tiled_matrix_desc_t *A,  tiled_matrix_desc_t *T,  tiled_matrix_desc_t *B, void *work);

/***********************************************************
 *               Destruct functions
 */
/* Level 2 Blas */
void dplasma_zgeadd_Destruct( parsec_handle_t *o );
void dplasma_zgerc_Destruct( parsec_handle_t *o );
void dplasma_zgeru_Destruct( parsec_handle_t *o );
void dplasma_ztradd_Destruct( parsec_handle_t *o );

/* Level 3 Blas */
void dplasma_zgemm_Destruct( parsec_handle_t *o );
void dplasma_zhemm_Destruct( parsec_handle_t *o );
void dplasma_zher2k_Destruct(parsec_handle_t *o );
void dplasma_zherk_Destruct( parsec_handle_t *o );
void dplasma_zsymm_Destruct( parsec_handle_t *o );
void dplasma_zsyr2k_Destruct(parsec_handle_t *o );
void dplasma_zsyrk_Destruct( parsec_handle_t *o );
void dplasma_ztrmm_Destruct( parsec_handle_t *o );
void dplasma_ztrsm_Destruct( parsec_handle_t *o );

/* Level 3 Blas extensions */
void dplasma_ztrdsm_Destruct( parsec_handle_t *o );
void dplasma_ztrsmpl_Destruct( parsec_handle_t *o );

/* Lapack */
void dplasma_zgelqf_Destruct( parsec_handle_t *o );
void dplasma_zgelqf_param_Destruct( parsec_handle_t *o );
void dplasma_zgeqrf_Destruct( parsec_handle_t *o );
void dplasma_zgeqrf_param_Destruct( parsec_handle_t *o );
void dplasma_zgetrf_Destruct( parsec_handle_t *o );
void dplasma_zgetrf_incpiv_Destruct( parsec_handle_t *o );
void dplasma_zgetrf_nopiv_Destruct( parsec_handle_t *o );
void dplasma_zlauum_Destruct( parsec_handle_t *o );
void dplasma_zpoinv_Destruct( parsec_handle_t *o );
void dplasma_zpotrf_Destruct( parsec_handle_t *o );
void dplasma_ztrtri_Destruct( parsec_handle_t *o );
void dplasma_zunglq_Destruct( parsec_handle_t *o );
void dplasma_zunglq_param_Destruct( parsec_handle_t *o );
void dplasma_zungqr_Destruct( parsec_handle_t *o );
void dplasma_zungqr_param_Destruct( parsec_handle_t *o );
void dplasma_zunmlq_Destruct( parsec_handle_t *o );
void dplasma_zunmlq_param_Destruct( parsec_handle_t *o );
void dplasma_zunmqr_Destruct( parsec_handle_t *o );
void dplasma_zunmqr_param_Destruct( parsec_handle_t *o );

/* Auxiliary routines */
void dplasma_zlacpy_Destruct( parsec_handle_t *o );
void dplasma_zlange_Destruct( parsec_handle_t *o );
void dplasma_zlanhe_Destruct( parsec_handle_t *o );
void dplasma_zlanm2_Destruct( parsec_handle_t *o );
void dplasma_zlansy_Destruct( parsec_handle_t *o );
void dplasma_zlantr_Destruct( parsec_handle_t *o );
void dplasma_zlascal_Destruct( parsec_handle_t *o );
void dplasma_zlaset_Destruct( parsec_handle_t *o );
void dplasma_zlaswp_Destruct( parsec_handle_t *o );
void dplasma_zplghe_Destruct( parsec_handle_t *o );
void dplasma_zplgsy_Destruct( parsec_handle_t *o );
void dplasma_zplrnt_Destruct( parsec_handle_t *o );

/* Sub-kernels to recursive DAGs */
void dplasma_zgeqrfr_geqrt_Destruct( parsec_handle_t *o );
void dplasma_zgeqrfr_tsmqr_Destruct( parsec_handle_t *o );
void dplasma_zgeqrfr_tsqrt_Destruct( parsec_handle_t *o );
void dplasma_zgeqrfr_unmqr_Destruct( parsec_handle_t *o );

/**********************************************************
 * Routines to set parameters in recursive DAGs
 */
void dplasma_zpotrf_setrecursive( parsec_handle_t *o, int hmb );
void dplasma_zgeqrf_setrecursive( parsec_handle_t *o, int hnb );

/**********************************************************
 * Check routines
 */
int check_zaxmb(  parsec_context_t *parsec, int loud, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *b, tiled_matrix_desc_t *x );
int check_zpotrf( parsec_context_t *parsec, int loud, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *A0 );
int check_zpoinv( parsec_context_t *parsec, int loud, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *Ainv );

/**********************************************************
 * Work in progress
 */

/* Hybrid LU-QR */
int  dplasma_zgetrf_qrf(  parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, int criteria, double alpha, int *lu_tab, int *INFO);
int  dplasma_ztrsmpl_qrf( parsec_context_t *parsec, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, int *lu_tab);

parsec_handle_t* dplasma_zgetrf_qrf_New( dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, int criteria, double alpha, int *lu_tab, int *INFO);
parsec_handle_t* dplasma_ztrsmpl_qrf_New(dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, int *lu_tab);

void dplasma_zgetrf_qrf_Destruct( parsec_handle_t *o );
void dplasma_ztrsmpl_qrf_Destruct( parsec_handle_t *o );

/* LDLt butterfly */
int dplasma_zhebut( parsec_context_t *parsec, tiled_matrix_desc_t *A, parsec_complex64_t **U_but_ptr, int level);
int dplasma_zhetrf(parsec_context_t *parsec, tiled_matrix_desc_t *A);
int dplasma_zhetrs( parsec_context_t *parsec, int uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, parsec_complex64_t *U_but_vec, int level);
int dplasma_ztrdsm( parsec_context_t *parsec, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

parsec_handle_t* dplasma_zgebmm_New( tiled_matrix_desc_t *A, parsec_complex64_t *U_but_vec, int it, int jt, int level, PLASMA_enum trans, int *info);
parsec_handle_t* dplasma_zgebut_New( tiled_matrix_desc_t *A, parsec_complex64_t *U_but_vec, int it, int jt, int level, int *info);
parsec_handle_t* dplasma_zhebut_New( tiled_matrix_desc_t *A, parsec_complex64_t *U_but_vec, int it, int jt, int level, int *info);
parsec_handle_t* dplasma_zhetrf_New( tiled_matrix_desc_t *A, int *info);
parsec_handle_t* dplasma_ztrdsm_New( const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B );
parsec_handle_t* dplasma_ztrmdm_New( tiled_matrix_desc_t *A);

void dplasma_zgebmm_Destruct( parsec_handle_t *o );
void dplasma_zgebut_Destruct( parsec_handle_t *o );
void dplasma_zhebut_Destruct( parsec_handle_t *o );
void dplasma_zhetrf_Destruct( parsec_handle_t *o );
void dplasma_ztrmdm_Destruct( parsec_handle_t *o );

/* SVD */
parsec_handle_t* dplasma_zhbrdt_New(tiled_matrix_desc_t *A);
parsec_handle_t* dplasma_zheev_New( const PLASMA_enum jobz, const PLASMA_enum uplo, tiled_matrix_desc_t* A, tiled_matrix_desc_t* W, tiled_matrix_desc_t* Z, int* info );
parsec_handle_t* dplasma_zherbt_New( PLASMA_enum uplo, int ib, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);

int dplasma_zheev( parsec_context_t *parsec, const PLASMA_enum jobz, const PLASMA_enum uplo, tiled_matrix_desc_t* A, tiled_matrix_desc_t* W, tiled_matrix_desc_t* Z );

void dplasma_zhbrdt_Destruct( parsec_handle_t *o );
void dplasma_zheev_Destruct( parsec_handle_t *o );
void dplasma_zherbt_Destruct( parsec_handle_t *o );


dague_handle_t*
dplasma_zgebrd_ge2gbx_New( int ib, dplasma_qrtree_t *qrtre0,
                           dplasma_qrtree_t *qrtree,
                           dplasma_qrtree_t *lqtree,
                           tiled_matrix_desc_t *A,
                           tiled_matrix_desc_t *TS0,
                           tiled_matrix_desc_t *TT0,
                           tiled_matrix_desc_t *TS,
                           tiled_matrix_desc_t *TT,
                           tiled_matrix_desc_t *Band );

void
dplasma_zgebrd_ge2gbx_Destruct( dague_handle_t *handle );

int
dplasma_zgebrd_ge2gbx( dague_context_t *dague,
                       int ib,
                       dplasma_qrtree_t *qrtre0,
                       dplasma_qrtree_t *qrtree,
                       dplasma_qrtree_t *lqtree,
                       tiled_matrix_desc_t *A,
                       tiled_matrix_desc_t *TS0,
                       tiled_matrix_desc_t *TT0,
                       tiled_matrix_desc_t *TS,
                       tiled_matrix_desc_t *TT,
                       tiled_matrix_desc_t *Band);

dague_handle_t* dplasma_zgebrd_ge2gb_New( int ib, tiled_matrix_desc_t *A, tiled_matrix_desc_t *Band );
void dplasma_zgebrd_ge2gb_Destruct( dague_handle_t *handle );
int dplasma_zgebrd_ge2gb( dague_context_t *dague, int ib, tiled_matrix_desc_t *A, tiled_matrix_desc_t *Band);

#endif /* _DPLASMA_Z_H_ */
