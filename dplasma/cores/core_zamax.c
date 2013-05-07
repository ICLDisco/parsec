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
                max      = abs < max ? max : abs;
                work[i]  = abs < max ? max : abs;
                tmpA++;
            }
            work[j] = max < cabs(*tmpA) ? cabs(*tmpA) : max;
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
                max      = abs < max ? max : abs;
                work[i]  = abs < max ? max : abs;
                tmpA++;
            }
            work[j] = max < work[j] ? work[j] : max;
        }
        break;
    case PlasmaUpperLower:
    default:
        if (storev == PlasmaColumnwise) {
            for (j = 0; j < N; j++) {
                tmpA = A+(j*lda);
                for (i = 0; i < M; i++) {
                    work[j] = work[j] < cabs(*tmpA) ? cabs(*tmpA) : work[j];
                    tmpA++;
                }
            }
        }
        else {
            for (j = 0; j < N; j++) {
                tmpA = A+(j*lda);
                tmpW = work;
                for (i = 0; i < M; i++) {
                    *tmpW = *tmpW < cabs(*tmpA) ? cabs(*tmpA) : *tmpW;
                    tmpA++; tmpW++;
                }
            }
        }
    }
    return PLASMA_SUCCESS;
}
