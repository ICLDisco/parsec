/*
 * Copyright (c) 2010-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 */
#ifndef _DPLASMA_Z_H_
#define _DPLASMA_Z_H_

#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

/***********************************************************
 *               Blocking interface
 */
/* Level 2 Blas */
int dplasma_zgerc( dague_context_t *dague, dague_complex64_t alpha,
                   const tiled_matrix_desc_t *X, const tiled_matrix_desc_t *Y, tiled_matrix_desc_t *A);
int dplasma_zgeru( dague_context_t *dague, dague_complex64_t alpha,
                   const tiled_matrix_desc_t *X, const tiled_matrix_desc_t *Y, tiled_matrix_desc_t *A);

/* Level 3 Blas */
int dplasma_zgemm( dague_context_t *dague, PLASMA_enum transA, PLASMA_enum transB,
                   dague_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                   dague_complex64_t beta,  tiled_matrix_desc_t *C);
int dplasma_zhemm( dague_context_t *dague, PLASMA_enum side, PLASMA_enum uplo,
                   dague_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                   dague_complex64_t beta,  tiled_matrix_desc_t *C);
int dplasma_zherk( dague_context_t *dague, PLASMA_enum uplo, PLASMA_enum trans,
                   double alpha, const tiled_matrix_desc_t *A,
                   double beta,  tiled_matrix_desc_t *C);
int dplasma_zher2k( dague_context_t *dague, PLASMA_enum uplo, PLASMA_enum trans,
                    dague_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                    double beta,  tiled_matrix_desc_t *C);
int dplasma_zsymm( dague_context_t *dague, PLASMA_enum side, PLASMA_enum uplo,
                   dague_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                   dague_complex64_t beta,  tiled_matrix_desc_t *C);
int dplasma_zsyrk( dague_context_t *dague, PLASMA_enum uplo, PLASMA_enum trans,
                   dague_complex64_t alpha, const tiled_matrix_desc_t *A,
                   dague_complex64_t beta,  tiled_matrix_desc_t *C);
int dplasma_zsyr2k( dague_context_t *dague, PLASMA_enum uplo, PLASMA_enum trans,
                    dague_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                    dague_complex64_t beta,  tiled_matrix_desc_t *C);
int dplasma_ztrmm( dague_context_t *dague, PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                   dague_complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
int dplasma_ztrsm( dague_context_t *dague, PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                   dague_complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

/* Level 3 Blas extensions */
int dplasma_ztrsmpl( dague_context_t *dague, const tiled_matrix_desc_t *A,
                     const tiled_matrix_desc_t *L, const tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B);

/* Lapack */
int    dplasma_zgelqf( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
int    dplasma_zgelqf_param( dague_context_t *dague, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT);
int    dplasma_zgelqs( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B );
int    dplasma_zgelqs_param( dague_context_t *dague, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B );
int    dplasma_zgels( dague_context_t *dague, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B );
int    dplasma_zgeqrf( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
int    dplasma_zgeqrf_param( dague_context_t *dague, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT);
int    dplasma_zgeqrf_rec( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, int hnb);
int    dplasma_zgeqrs( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B );
int    dplasma_zgeqrs_param( dague_context_t *dague, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B );
int    dplasma_zgesv ( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zgesv_incpiv ( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zgetrf( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV );
int    dplasma_zgetrf_incpiv( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV );
int    dplasma_zgetrf_nopiv(  dague_context_t *dague, tiled_matrix_desc_t *A );
int    dplasma_zgetrs( dague_context_t *dague, const PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zgetrs_incpiv( dague_context_t *dague, const PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B );
int    dplasma_zlauum( dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A );
int    dplasma_zpoinv( dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A);
int    dplasma_zpoinv_sync( dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A);
int    dplasma_zposv ( dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
int    dplasma_zpotrf( dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A);
int    dplasma_zpotrf_rec( dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A, int hmb );
int    dplasma_zpotrs( dague_context_t *dague, PLASMA_enum uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
int    dplasma_zpotri( dague_context_t *dague, PLASMA_enum uplo, tiled_matrix_desc_t *A);
int    dplasma_ztrtri( dague_context_t *dague, PLASMA_enum uplo, PLASMA_enum diag, tiled_matrix_desc_t *A );
int    dplasma_zunglq( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *Q);
int    dplasma_zunglq_param( dague_context_t *dague, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *Q);
int    dplasma_zungqr( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *Q);
int    dplasma_zungqr_param( dague_context_t *dague, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *Q);
int    dplasma_zunmlq( dague_context_t *dague, PLASMA_enum side, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B);
int    dplasma_zunmlq_param( dague_context_t *dague, PLASMA_enum side, PLASMA_enum trans, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B );
int    dplasma_zunmqr( dague_context_t *dague, PLASMA_enum side, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B);
int    dplasma_zunmqr_param( dague_context_t *dague, PLASMA_enum side, PLASMA_enum trans, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B );

/* Lapack Auxiliary */
int    dplasma_zgeadd( dague_context_t *dague, PLASMA_enum transA, PLASMA_enum uplo, dague_complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
int    dplasma_zlacpy( dague_context_t *dague, PLASMA_enum uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
double dplasma_zlange( dague_context_t *dague, PLASMA_enum ntype, const tiled_matrix_desc_t *A);
double dplasma_zlanhe( dague_context_t *dague, PLASMA_enum ntype, PLASMA_enum uplo, const tiled_matrix_desc_t *A);
double dplasma_zlansy( dague_context_t *dague, PLASMA_enum ntype, PLASMA_enum uplo, const tiled_matrix_desc_t *A);
double dplasma_zlantr( dague_context_t *dague, PLASMA_enum ntype, PLASMA_enum uplo, PLASMA_enum diag, const tiled_matrix_desc_t *A);
int    dplasma_zlascal(dague_context_t *dague, PLASMA_enum type, dague_complex64_t alpha, tiled_matrix_desc_t *A);
int    dplasma_zlaset( dague_context_t *dague, PLASMA_enum uplo, dague_complex64_t alpha, dague_complex64_t beta, tiled_matrix_desc_t *A);
int    dplasma_zlaswp( dague_context_t *dague, tiled_matrix_desc_t *A, const tiled_matrix_desc_t *IPIV, int inc);
int    dplasma_zplghe( dague_context_t *dague, double            bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
int    dplasma_zplgsy( dague_context_t *dague, dague_complex64_t bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
int    dplasma_zplrnt( dague_context_t *dague, int diagdom,                              tiled_matrix_desc_t *A, unsigned long long int seed);

/* Auxiliary routines available only through synchronous interface */
int    dplasma_zpltmg( dague_context_t *dague, PLASMA_enum mtxtype, tiled_matrix_desc_t *A, unsigned long long int seed);
int    dplasma_zprint( dague_context_t *dague, PLASMA_enum uplo, const tiled_matrix_desc_t *A);

/***********************************************************
 *             Non-Blocking interface
 */
/* Level 2 Blas */
dague_handle_t* dplasma_zgerc_New( dague_complex64_t alpha, const tiled_matrix_desc_t *X, const tiled_matrix_desc_t *Y, tiled_matrix_desc_t *A);
dague_handle_t* dplasma_zgeru_New( dague_complex64_t alpha, const tiled_matrix_desc_t *X, const tiled_matrix_desc_t *Y, tiled_matrix_desc_t *A);

/* Level 3 Blas */
dague_handle_t* dplasma_zgemm_New( PLASMA_enum transa, PLASMA_enum transb,
                                   dague_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                                   dague_complex64_t beta,  tiled_matrix_desc_t *C);
dague_handle_t* dplasma_zhemm_New( PLASMA_enum side, PLASMA_enum uplo,
                                   dague_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                                   dague_complex64_t beta,  tiled_matrix_desc_t *C);
dague_handle_t* dplasma_zherk_New( PLASMA_enum uplo, PLASMA_enum trans,
                                   double alpha, const tiled_matrix_desc_t *A,
                                   double beta,  tiled_matrix_desc_t *C);
dague_handle_t* dplasma_zher2k_New( PLASMA_enum uplo, PLASMA_enum trans,
                                    dague_complex64_t alpha, const tiled_matrix_desc_t *A,
                                    const tiled_matrix_desc_t *B,
                                    double beta,  tiled_matrix_desc_t *C);
dague_handle_t* dplasma_zsymm_New( PLASMA_enum side, PLASMA_enum uplo,
                                   dague_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
                                   dague_complex64_t beta,  tiled_matrix_desc_t *C);
dague_handle_t* dplasma_zsyrk_New( PLASMA_enum uplo, PLASMA_enum trans,
                                   dague_complex64_t alpha, const tiled_matrix_desc_t *A,
                                   dague_complex64_t beta,  tiled_matrix_desc_t *C);
dague_handle_t* dplasma_zsyr2k_New( PLASMA_enum uplo, PLASMA_enum trans,
                                    dague_complex64_t alpha, const tiled_matrix_desc_t *A,
                                    const tiled_matrix_desc_t *B,
                                    dague_complex64_t beta,  tiled_matrix_desc_t *C);
dague_handle_t* dplasma_ztrmm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                                   dague_complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
dague_handle_t* dplasma_ztrsm_New( PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum trans, PLASMA_enum diag,
                                   const dague_complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

/* Level 3 Blas extensions */
dague_handle_t* dplasma_ztrsmpl_New(const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *L,
                                    const tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B);

/* Lapack */
dague_handle_t* dplasma_zgelqf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
dague_handle_t* dplasma_zgelqf_param_New(dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT);
dague_handle_t* dplasma_zgeqrf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);
dague_handle_t* dplasma_zgeqrf_param_New(dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT);
dague_handle_t* dplasma_zgetrf_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, int *INFO);
dague_handle_t* dplasma_zgetrf_incpiv_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, tiled_matrix_desc_t *IPIV, int *INFO);
dague_handle_t* dplasma_zgetrf_nopiv_New(tiled_matrix_desc_t *A, int *INFO);
dague_handle_t* dplasma_zlauum_New( PLASMA_enum uplo, tiled_matrix_desc_t *A);
dague_handle_t* dplasma_zpoinv_New( PLASMA_enum uplo, tiled_matrix_desc_t *A, int *INFO);
dague_handle_t* dplasma_zpotrf_New( PLASMA_enum uplo, tiled_matrix_desc_t *A, int *INFO);
dague_handle_t* dplasma_ztrtri_New( PLASMA_enum uplo, PLASMA_enum diag, tiled_matrix_desc_t *A, int *info);
dague_handle_t* dplasma_zunglq_New( tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *Q);
dague_handle_t* dplasma_zunglq_param_New(dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *Q);
dague_handle_t* dplasma_zungqr_New( tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *Q);
dague_handle_t* dplasma_zungqr_param_New(dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *Q);
dague_handle_t* dplasma_zunmlq_New( PLASMA_enum side, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B);
dague_handle_t* dplasma_zunmlq_param_New( PLASMA_enum side, PLASMA_enum trans, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B);
dague_handle_t* dplasma_zunmqr_New( PLASMA_enum side, PLASMA_enum trans, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *B);
dague_handle_t* dplasma_zunmqr_param_New( PLASMA_enum side, PLASMA_enum trans, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, tiled_matrix_desc_t *B);

/* Auxiliary routines */
dague_handle_t* dplasma_zgeadd_New( PLASMA_enum transA, PLASMA_enum uplo, dague_complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
dague_handle_t* dplasma_zlacpy_New( PLASMA_enum uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);
dague_handle_t* dplasma_zlange_New( PLASMA_enum ntype, const tiled_matrix_desc_t *A, double *norm);
dague_handle_t* dplasma_zlanhe_New( PLASMA_enum ntype, PLASMA_enum uplo, const tiled_matrix_desc_t *A, double *result);
dague_handle_t* dplasma_zlansy_New( PLASMA_enum ntype, PLASMA_enum uplo, const tiled_matrix_desc_t *A, double *result);
dague_handle_t* dplasma_zlantr_New( PLASMA_enum ntype, PLASMA_enum uplo, PLASMA_enum diag, const tiled_matrix_desc_t *A, double *result);
dague_handle_t* dplasma_zlascal_New( PLASMA_enum type, dague_complex64_t alpha, tiled_matrix_desc_t *A);
dague_handle_t* dplasma_zlaset_New( PLASMA_enum uplo, dague_complex64_t alpha, dague_complex64_t beta, tiled_matrix_desc_t *A);
dague_handle_t* dplasma_zlaswp_New( tiled_matrix_desc_t *A, const tiled_matrix_desc_t *IPIV, int inc);
dague_handle_t* dplasma_zplghe_New( double            bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
dague_handle_t* dplasma_zplgsy_New( dague_complex64_t bump, PLASMA_enum uplo, tiled_matrix_desc_t *A, unsigned long long int seed);
dague_handle_t* dplasma_zplrnt_New( int diagdom,                              tiled_matrix_desc_t *A, unsigned long long int seed);

/***********************************************************
 *               Destruct functions
 */
/* Level 2 Blas */
void dplasma_zgerc_Destruct( dague_handle_t *o );
void dplasma_zgeru_Destruct( dague_handle_t *o );

/* Level 3 Blas */
void dplasma_zgemm_Destruct( dague_handle_t *o );
void dplasma_zhemm_Destruct( dague_handle_t *o );
void dplasma_zher2k_Destruct(dague_handle_t *o );
void dplasma_zherk_Destruct( dague_handle_t *o );
void dplasma_zsymm_Destruct( dague_handle_t *o );
void dplasma_zsyr2k_Destruct(dague_handle_t *o );
void dplasma_zsyrk_Destruct( dague_handle_t *o );
void dplasma_ztrmm_Destruct( dague_handle_t *o );
void dplasma_ztrsm_Destruct( dague_handle_t *o );

/* Level 3 Blas extensions */
void dplasma_ztrdsm_Destruct( dague_handle_t *o );
void dplasma_ztrsmpl_Destruct( dague_handle_t *o );

/* Lapack */
void dplasma_zgelqf_Destruct( dague_handle_t *o );
void dplasma_zgelqf_param_Destruct( dague_handle_t *o );
void dplasma_zgeqrf_Destruct( dague_handle_t *o );
void dplasma_zgeqrf_param_Destruct( dague_handle_t *o );
void dplasma_zgetrf_Destruct( dague_handle_t *o );
void dplasma_zgetrf_incpiv_Destruct( dague_handle_t *o );
void dplasma_zgetrf_nopiv_Destruct( dague_handle_t *o );
void dplasma_zlauum_Destruct( dague_handle_t *o );
void dplasma_zpoinv_Destruct( dague_handle_t *o );
void dplasma_zpotrf_Destruct( dague_handle_t *o );
void dplasma_ztrtri_Destruct( dague_handle_t *o );
void dplasma_zunglq_Destruct( dague_handle_t *o );
void dplasma_zunglq_param_Destruct( dague_handle_t *o );
void dplasma_zungqr_Destruct( dague_handle_t *o );
void dplasma_zungqr_param_Destruct( dague_handle_t *o );
void dplasma_zunmlq_Destruct( dague_handle_t *o );
void dplasma_zunmlq_param_Destruct( dague_handle_t *o );
void dplasma_zunmqr_Destruct( dague_handle_t *o );
void dplasma_zunmqr_param_Destruct( dague_handle_t *o );

/* Auxiliary routines */
void dplasma_zgeadd_Destruct( PLASMA_enum transA, dague_handle_t *o );
void dplasma_zlacpy_Destruct( dague_handle_t *o );
void dplasma_zlange_Destruct( dague_handle_t *o );
void dplasma_zlanhe_Destruct( dague_handle_t *o );
void dplasma_zlansy_Destruct( dague_handle_t *o );
void dplasma_zlantr_Destruct( dague_handle_t *o );
void dplasma_zlascal_Destruct( dague_handle_t *o );
void dplasma_zlaset_Destruct( dague_handle_t *o );
void dplasma_zlaswp_Destruct( dague_handle_t *o );
void dplasma_zplghe_Destruct( dague_handle_t *o );
void dplasma_zplgsy_Destruct( dague_handle_t *o );
void dplasma_zplrnt_Destruct( dague_handle_t *o );

/**********************************************************
 * Routines to set parameters in recursive DAGs
 */
void dplasma_zpotrf_setrecursive( dague_handle_t *o, int hmb );
void dplasma_zgeqrf_setrecursive( dague_handle_t *o, int hnb );

/**********************************************************
 * Check routines
 */
int check_zaxmb(  dague_context_t *dague, int loud, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *b, tiled_matrix_desc_t *x );
int check_zpotrf( dague_context_t *dague, int loud, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *A0 );
int check_zpoinv( dague_context_t *dague, int loud, PLASMA_enum uplo, tiled_matrix_desc_t *A, tiled_matrix_desc_t *Ainv );

/**********************************************************
 * Work in progress
 */

/* Hybrid LU-QR */
int  dplasma_zgetrf_qrf(  dague_context_t *dague, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, int criteria, double alpha, int *lu_tab, int *INFO);
int  dplasma_ztrsmpl_qrf( dague_context_t *dague, dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, int *lu_tab);

dague_handle_t* dplasma_zgetrf_qrf_New( dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, int criteria, double alpha, int *lu_tab, int *INFO);
dague_handle_t* dplasma_ztrsmpl_qrf_New(dplasma_qrtree_t *qrtree, tiled_matrix_desc_t *A, tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B, tiled_matrix_desc_t *TS, tiled_matrix_desc_t *TT, int *lu_tab);

void dplasma_zgetrf_qrf_Destruct( dague_handle_t *o );
void dplasma_ztrsmpl_qrf_Destruct( dague_handle_t *o );

/* LDLt butterfly */
int dplasma_zhebut( dague_context_t *dague, tiled_matrix_desc_t *A, dague_complex64_t **U_but_ptr, int level);
int dplasma_zhetrf(dague_context_t *dague, tiled_matrix_desc_t *A);
int dplasma_zhetrs( dague_context_t *dague, int uplo, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, dague_complex64_t *U_but_vec, int level);
int dplasma_ztrdsm( dague_context_t *dague, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B);

dague_handle_t* dplasma_zgebmm_New( tiled_matrix_desc_t *A, dague_complex64_t *U_but_vec, int it, int jt, int level, PLASMA_enum trans, int *info);
dague_handle_t* dplasma_zgebut_New( tiled_matrix_desc_t *A, dague_complex64_t *U_but_vec, int it, int jt, int level, int *info);
dague_handle_t* dplasma_zhebut_New( tiled_matrix_desc_t *A, dague_complex64_t *U_but_vec, int it, int jt, int level, int *info);
dague_handle_t* dplasma_zhetrf_New( tiled_matrix_desc_t *A, int *info);
dague_handle_t* dplasma_ztrdsm_New( const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B );
dague_handle_t* dplasma_ztrmdm_New( tiled_matrix_desc_t *A);

void dplasma_zgebmm_Destruct( dague_handle_t *o );
void dplasma_zgebut_Destruct( dague_handle_t *o );
void dplasma_zhebut_Destruct( dague_handle_t *o );
void dplasma_zhetrf_Destruct( dague_handle_t *o );
void dplasma_ztrmdm_Destruct( dague_handle_t *o );

/* SVD */
dague_handle_t* dplasma_zhbrdt_New(tiled_matrix_desc_t *A);
dague_handle_t* dplasma_zheev_New( const PLASMA_enum jobz, const PLASMA_enum uplo, tiled_matrix_desc_t* A, tiled_matrix_desc_t* W, tiled_matrix_desc_t* Z, int* info );
dague_handle_t* dplasma_zherbt_New( PLASMA_enum uplo, int ib, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T);

int dplasma_zheev( dague_context_t *dague, const PLASMA_enum jobz, const PLASMA_enum uplo, tiled_matrix_desc_t* A, tiled_matrix_desc_t* W, tiled_matrix_desc_t* Z );

void dplasma_zhbrdt_Destruct( dague_handle_t *o );
void dplasma_zheev_Destruct( dague_handle_t *o );
void dplasma_zherbt_Destruct( dague_handle_t *o );

#endif /* _DPLASMA_Z_H_ */
