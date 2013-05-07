/**
 *
 * @file core_dzamax.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.5.0
 * @author Mathieu Faverge
 * @author Julien Herrmann
 * @date 2013-03-12
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

#ifdef BLKLDD
#undef BLKLDD
#define BLKLDD(A, k) ( ( (k) + (A).i/(A).mb) < (A).lm1 ? (A).mb : (A).lm%(A).mb )
#endif

/***************************************************************************//**
 *  Internal function to return adress of block (m,n)
 **/
inline static void *plasma_getaddr(PLASMA_desc A, int m, int n)
{
    int mm = m+A.i/A.mb;
    int nn = n+A.j/A.nb;
    size_t eltsize = sizeof( PLASMA_Complex64_t );
    size_t offset = 0;

    if (mm < A.lm1) {
        if (nn < A.ln1)
            offset = A.bsiz*(mm+A.lm1*nn);
        else
            offset = A.A12 + (A.mb*(A.ln%A.nb)*mm);
    }
    else {
        if (nn < A.ln1)
            offset = A.A21 + ((A.lm%A.mb)*A.nb*nn);
        else
            offset = A.A22;
    }

    return (void*)((intptr_t)A.mat + (offset*eltsize) );
}

#define BLKADDR(A, type, m, n)  (type *)plasma_getaddr(A, m, n)
#define A(m) BLKADDR(descA, PLASMA_Complex64_t, m, 0)

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 **/
#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_dzamax = PCORE_dzamax
#define CORE_dzamax PCORE_dzamax
#endif
int CORE_zamax(PLASMA_enum storev, PLASMA_enum uplo, int M, int N,
               const PLASMA_Complex64_t *A, int lda, double *work)
{
    const PLASMA_Complex64_t *tmpA;
    double *tmpW, max, abs;
    int i,j;

    switch (uplo) {
    case PlasmaUpper:
        for (j = 0; j < N; j++) {
            tmpA = A+(j*lda);
            max = 0.0;
            for (i = 0; i < j; i++) {
                abs      = cabs(*tmpA);
                max      = abs < max     ? max     : abs;
                work[i]  = abs < work[i] ? work[i] : abs;
                tmpA++;
            }
            work[j] = work[j] < max         ? max         : work[j];
            work[j] = work[j] < cabs(*tmpA) ? cabs(*tmpA) : work[j];
        }
        break;
    case PlasmaLower:
        for (j = 0; j < N; j++) {
            tmpA = A+(j*lda)+j;

            max = 0.0;
            work[j] = work[j] < cabs(*tmpA) ? cabs(*tmpA) : work[j];

            tmpA++;
            for (i = j+1; i < M; i++) {
                abs      = cabs(*tmpA);
                max      = abs < max     ? max     : abs;
                work[i]  = abs < work[i] ? work[i] : abs;
                tmpA++;
            }
            work[j] = work[j] < max ? max : work[j];
        }
        break;
    case PlasmaUpperLower:
    default:
        if (storev == PlasmaColumnwise) {
            for (j = 0; j < N; j++) {
                tmpA = A+(j*lda);
                for (i = 0; i < M; i++) {
                    abs = cabs(*tmpA);
                    work[j] = work[j] < abs ? abs : work[j];
                    tmpA++;
                }
            }
        }
        else {
            for (j = 0; j < N; j++) {
                tmpA = A+(j*lda);
                tmpW = work;
                for (i = 0; i < M; i++) {
                    abs = cabs(*tmpA);
                    *tmpW = *tmpW < abs ? abs : *tmpW;
                    tmpA++; tmpW++;
                }
            }
        }
    }
    return PLASMA_SUCCESS;
}


#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_dzamax_tile = PCORE_dzamax_tile
#define CORE_dzamax_tile PCORE_dzamax_tile
#endif
int CORE_zamax_tile( PLASMA_enum storev,
                     PLASMA_enum uplo,
                     const PLASMA_desc descA,
                     double *work)
{
    PLASMA_Complex64_t *A;
    int m, lda, tempmm;

    /* Check input arguments */
    if ((storev != PlasmaColumnwise) && (storev != PlasmaRowwise)) {
        coreblas_error(1, "Illegal value of storev");
        return -1;
    }
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper) &&
        (uplo != PlasmaUpperLower))
    {
        coreblas_error(2, "Illegal value of uplo");
        return -2;
    }
    if (storev != PlasmaColumnwise) {
        coreblas_error(1, "Illegal value of storev (Not supported)");
        return -1;
    }
    if (uplo != PlasmaUpperLower)
    {
        coreblas_error(2, "Illegal value of uplo (Not supported)");
        return -2;
    }

    for( m = 0; m < descA.mt; m++)
    {
        lda    = BLKLDD( descA, m );
        A      = A( m );
        tempmm = (m == (descA.mt-1)) ? (descA.m - m * descA.mb) : descA.mb;

        /* Apply the GEMM */
        CORE_zamax( PlasmaColumnwise, PlasmaUpperLower,
                    tempmm, descA.n,
                    A, lda, work );
    }

    return PLASMA_SUCCESS;
}
