/**
 *
 * @file core_zplssq.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.4.6
 * @author Mathieu Faverge
 * @date 2010-11-15
 * @precisions normal z -> c d s
 *
 **/
#include "dplasma_zcores.h"
#include <math.h>
#include <lapacke.h>
//#include "common.h"
#include <plasma.h>
#include "cblas.h"
#include "core_blas.h"

#define COMPLEX

/*****************************************************************************
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  CORE_zplssq returns the values scl and ssq such that
 *
 *    ( scl**2 )*ssq = x( 1 )**2 +...+ x( n )**2 + ( scale**2 )*sumsq,
 *
 * where x( i ) = abs( X( 1 + ( i - 1 )*INCX ) ). The value of sumsq is
 * assumed to be at least unity and the value of ssq will then satisfy
 *
 *    1.0 .le. ssq .le. ( sumsq + 2*n ).
 *
 * scale is assumed to be non-negative and scl returns the value
 *
 *    scl = max( scale, abs( real( x( i ) ) ), abs( aimag( x( i ) ) ) ),
 *           i
 *
 * scale and sumsq must be supplied in SCALE and SUMSQ respectively.
 * SCALE and SUMSQ are overwritten by scl and ssq respectively.
 *
 * The routine makes only one pass through the tile A.
 *
 *******************************************************************************
 *
 *  @param[in] M
 *          The number of rows in the tile A.
 *
 *  @param[in] N
 *          The number of columns in the tile A.
 *
 *  @param[in] A
 *          The M-by-N matrix on which to compute the norm.
 *
 *  @param[in] LDA
 *          The leading dimension of the tile A. LDA >= max(1,M).
 *
 *  @param[in,out] scale
 *          On entry, the value  scale  in the equation above.
 *          On exit, scale is overwritten with the value scl.
 *
 *  @param[in,out] sumsq
 *          On entry, the value  sumsq  in the equation above.
 *          On exit, SUMSQ is overwritten with the value ssq.
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval -k, the k-th argument had an illegal value
 *
 */
#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_zplssq = PCORE_zplssq
#define CORE_zplssq PCORE_zplssq
#endif
int CORE_zplssq(int M, int N,
                PLASMA_Complex64_t *A, int LDA,
                double *scale, double *sumsq)
{
    int i, j;
    double tmp;
    double *ptr;

    for(j=0; j<N; j++) {
        ptr = (double*) ( A + j * LDA );
        for(i=0; i<M; i++, ptr++) {
            tmp = fabs(*ptr);
            if ( *scale < tmp ) {
                *sumsq = 1 + (*sumsq) * ( *scale / tmp ) * ( *scale / tmp );
                *scale = tmp;
            } else {
                *sumsq = *sumsq + ( tmp / *scale ) *  ( tmp / *scale );
            }
#ifdef COMPLEX
            ptr++;
            tmp = fabs(*ptr);
            if ( *scale < tmp ) {
                *sumsq = 1 + *sumsq * ( *scale / tmp ) * ( *scale / tmp );
                *scale = tmp;
            } else {
                *sumsq = *sumsq + ( tmp / *scale ) * ( tmp / *scale );
            }
#endif
        }
    }
    return PLASMA_SUCCESS;
}
