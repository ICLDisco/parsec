/**
 *
 * @file dplasma_zf77.c
 *
 *  DPLASMA Fortran 77 interface for computational routines
 *  DPLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 1.0.0
 * @author Mathieu Faverge
 * @date 2011-12-05
 **/
/*
 * @precisions normal z -> c d s
 */
#include "parsec.h"
#include <core_blas.h>
#include "dplasma.h"
#include "dplasmaf77.h"
#include "data_dist/matrix/matrix.h"

#define dplasmaf77_zgemm      DPLASMA_ZF77NAME( gemm,      GEMM      )
#define dplasmaf77_zhemm      DPLASMA_ZF77NAME( hemm,      HEMM      )
#define dplasmaf77_ztrmm      DPLASMA_ZF77NAME( trmm,      TRMM      )
#define dplasmaf77_ztrsm      DPLASMA_ZF77NAME( trsm,      TRSM      )
#define dplasmaf77_ztrsmpl    DPLASMA_ZF77NAME( trsmpl,    TRSMPL    )
#define dplasmaf77_ztrsmpl_sd DPLASMA_ZF77NAME( trsmpl_sd, TRSMPL_SD )

/* Lapack */
#define dplasmaf77_zpotrf       DPLASMA_ZF77NAME( potrf,       POTRF       )
#define dplasmaf77_zpotrs       DPLASMA_ZF77NAME( potrs,       POTRS       )
#define dplasmaf77_zposv        DPLASMA_ZF77NAME( posv,        POSV        )
#define dplasmaf77_zgetrf       DPLASMA_ZF77NAME( getrf,       GETRF       )
#define dplasmaf77_zgetrs       DPLASMA_ZF77NAME( getrs,       GETRS       )
#define dplasmaf77_zgesv        DPLASMA_ZF77NAME( gesv,        GESV        )
#define dplasmaf77_zgeqrf       DPLASMA_ZF77NAME( geqrf,       GEQRF       )
#define dplasmaf77_zgeqrf_param DPLASMA_ZF77NAME( geqrf_param, GEQRF_PARAM )
#define dplasmaf77_zgelqf       DPLASMA_ZF77NAME( gelqf,       GELQF       )
#define dplasmaf77_zungqr       DPLASMA_ZF77NAME( ungqr,       UNGQR       )
#define dplasmaf77_zungqr_param DPLASMA_ZF77NAME( ungqr_param, UNGQR_PARAM )

#define dplasmaf77_zgeadd       DPLASMA_ZF77NAME( geadd, GEADD )
#define dplasmaf77_zlacpy       DPLASMA_ZF77NAME( lacpy, LACPY )
#define dplasmaf77_zlaset       DPLASMA_ZF77NAME( laset, LASET )
#if defined(PRECISION_z) || defined(PRECISION_c)
#define dplasmaf77_zplghe       DPLASMA_ZF77NAME( plghe, PLGHE )
#endif
#define dplasmaf77_zplgsy       DPLASMA_ZF77NAME( plgsy, PLGSY )
#define dplasmaf77_zplrnt       DPLASMA_ZF77NAME( plrnt, PLRNT )

#define dplasmaf77_zlange       DPLASMA_ZF77NAME( lange, LANGE )
#define dplasmaf77_zlanhe       DPLASMA_ZF77NAME( lanhe, LANHE )

void dplasmaf77_zgemm( int *transA, int *transB, parsec_complex64_t *alpha, parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **B, parsec_complex64_t *beta, parsec_tiled_matrix_dc_t **C)
{
    extern parsec_context_t *parsecf77_context;
    dplasma_zgemm( parsecf77_context, *transA, *transB, *alpha, *A, *B, *beta, *C) ;
}


void dplasmaf77_zhemm( PLASMA_enum *side, PLASMA_enum *uplo, parsec_complex64_t *alpha, parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **B, double *beta, parsec_tiled_matrix_dc_t **C)
{
    extern parsec_context_t *parsecf77_context;
    dplasma_zhemm( parsecf77_context, *side, *uplo, *alpha, *A, *B, *beta, *C) ;
}

void dplasmaf77_ztrmm( PLASMA_enum *side, PLASMA_enum *uplo, PLASMA_enum *trans, PLASMA_enum *diag, PLASMA_Complex64_t *alpha, parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **B)
{
    extern parsec_context_t *parsecf77_context;
    dplasma_ztrmm( parsecf77_context, *side, *uplo, *trans, *diag, *alpha, *A, *B) ;
}

void dplasmaf77_ztrsm( PLASMA_enum *side, PLASMA_enum *uplo, PLASMA_enum *trans, PLASMA_enum *diag, PLASMA_Complex64_t *alpha, parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **B)
{
    extern parsec_context_t *parsecf77_context;
    dplasma_ztrsm( parsecf77_context, *side, *uplo, *trans, *diag, *alpha, *A, *B) ;
}

void dplasmaf77_ztrsmpl( parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **L, parsec_tiled_matrix_dc_t **IPIV, parsec_tiled_matrix_dc_t **B)
{
    extern parsec_context_t *parsecf77_context;
    dplasma_ztrsmpl( parsecf77_context, *A, *L, *IPIV, *B) ;
}


/* Lapack */
void dplasmaf77_zpotrf( PLASMA_enum *uplo, parsec_tiled_matrix_dc_t **A, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zpotrf( parsecf77_context, *uplo, *A) ;

}

void dplasmaf77_zpotrs( PLASMA_enum *uplo, parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **B, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zpotrs( parsecf77_context, *uplo, *A, *B) ;

}

void dplasmaf77_zposv ( PLASMA_enum *uplo, parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **B, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zposv( parsecf77_context, *uplo, *A, *B) ;

}

void dplasmaf77_zgetrf( parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **L, parsec_tiled_matrix_dc_t **IPIV, int *ret ) 
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zgetrf( parsecf77_context, *A, *L, *IPIV)  ;

}

void dplasmaf77_zgetrs( PLASMA_enum *trans, parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **L, parsec_tiled_matrix_dc_t **IPIV, parsec_tiled_matrix_dc_t **B, int *ret ) 
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zgetrs( parsecf77_context, *trans, *A, *L, *IPIV, *B)  ;

}

void dplasmaf77_zgesv ( parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **L, parsec_tiled_matrix_dc_t **IPIV, parsec_tiled_matrix_dc_t **B, int *ret ) 
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zgesv( parsecf77_context, *A, *L, *IPIV, *B)  ;

}

void dplasmaf77_zgeqrf( parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **T, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zgeqrf( parsecf77_context, *A, *T) ;

}

void dplasmaf77_zgeqrf_param( qr_piv_t **qrpiv, parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **TS, parsec_tiled_matrix_dc_t **TT, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zgeqrf_param( parsecf77_context, *qrpiv, *A, *TS, *TT) ;

}

void dplasmaf77_zgelqf( parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **T, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zgelqf( parsecf77_context, *A, *T) ;

}

void dplasmaf77_zungqr( parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **T, parsec_tiled_matrix_dc_t **Q, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zungqr( parsecf77_context, *A, *T, *Q) ;

}

void dplasmaf77_zungqr_param( qr_piv_t **qrpiv, parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **TS, parsec_tiled_matrix_dc_t **TT, parsec_tiled_matrix_dc_t **Q, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zungqr_param( parsecf77_context, *qrpiv, *A, *TS, *TT, *Q) ;

}


void dplasmaf77_zgeadd( PLASMA_enum *uplo, parsec_complex64_t *alpha, parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **B, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zgeadd( parsecf77_context, *uplo, *alpha, *A, *B) ;

}

void dplasmaf77_zlacpy( PLASMA_enum *uplo, parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **B, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zlacpy( parsecf77_context, *uplo, *A, *B) ;

}

void dplasmaf77_zlaset( PLASMA_enum *uplo, parsec_complex64_t *alpha, parsec_complex64_t *beta, parsec_tiled_matrix_dc_t **A, int *ret ) 
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zlaset( parsecf77_context, *uplo, *alpha, *beta, *A)  ;

}

#if defined(PRECISION_z) || defined(PRECISION_c)
void dplasmaf77_zplghe( double *bump, PLASMA_enum *uplo, parsec_tiled_matrix_dc_t **A, unsigned long long int *seed, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zplghe( parsecf77_context, *bump, *uplo, *A, *seed) ;

}

#endif
void dplasmaf77_zplgsy( parsec_complex64_t *bump, PLASMA_enum *uplo, parsec_tiled_matrix_dc_t **A, unsigned long long int *seed, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zplgsy( parsecf77_context, *bump, *uplo, *A, *seed) ;

}

void dplasmaf77_zplrnt( parsec_tiled_matrix_dc_t **A, unsigned long long int *seed, int *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zplrnt( parsecf77_context, *A, *seed) ;

}


void dplasmaf77_zlange( PLASMA_enum *ntype, parsec_tiled_matrix_dc_t **A, double *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zlange( parsecf77_context, *ntype, *A) ;

}

void dplasmaf77_zlanhe( PLASMA_enum *ntype, PLASMA_enum *uplo, parsec_tiled_matrix_dc_t **A, double *ret )
{
    extern parsec_context_t *parsecf77_context;
    *ret = dplasma_zlanhe( parsecf77_context, *ntype, *uplo, *A) ;

}


void dplasmaf77_ztrsmpl_sd( parsec_tiled_matrix_dc_t **A, parsec_tiled_matrix_dc_t **L, parsec_tiled_matrix_dc_t **B)
{
    extern parsec_context_t *parsecf77_context;
    dplasma_ztrsmpl_sd( parsecf77_context, *A, *L, *B) ;
}

