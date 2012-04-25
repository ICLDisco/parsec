/**
 *
 * @file core_zhetrf2_nopiv.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.4.5
 * @author Dulceneia Becker
 * @date 2011-1-18
 * @precisions normal z -> c d s
 *
 **/
#include <lapacke.h>
#include "dague_config.h"
#include "dplasma_cores.h"
#include "dplasma_zcores.h"

#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */
#if defined(HAVE_STDARG_H)
#include <stdarg.h>
#endif  /* defined(HAVE_STDARG_H) */
#include <stdio.h>
#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif

#include "plasma.h"
#include "cblas.h"
#include "core_blas.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

extern void CORE_zhetrf_nopiv(int uplo, int N, int ib,
                         PLASMA_Complex64_t *A, int LDA,
                         PLASMA_Complex64_t *WORK, int LDWORK,
                         int *INFO);

void CORE_zhetrf2_nopiv(PLASMA_enum uplo, int N, int ib,
        PLASMA_Complex64_t *A, int LDA,
        PLASMA_Complex64_t *WORK, int LWORK, int *INFO);

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 * CORE_zhetrf2_nopiv: ZHETRF_NOPIV followed by L*D
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_zhetrf2_nopiv = PCORE_zhetrf2_nopiv
#define CORE_zhetrf2_nopiv PCORE_zhetrf2_nopiv
#endif
void CORE_zhetrf2_nopiv(PLASMA_enum uplo, int N, int ib,
        PLASMA_Complex64_t *A, int LDA,
        PLASMA_Complex64_t *WORK, int LWORK, int *INFO)
{

    int j;
    PLASMA_Complex64_t alpha;

    /* Factorize A as L*D*L' using the lower/upper triangle of A */
    CORE_zhetrf_nopiv(uplo, N, ib, A, LDA, WORK, LWORK, INFO);

    if (uplo==PlasmaLower) {

        // Multiply L by D
        for (j=0; j<(N-1); j++) {
            alpha = A[LDA*j+j];
            cblas_zscal(N-j-1, CBLAS_SADDR(alpha), &A[LDA*j+j+1], 1);
        }

    } else if (uplo==PlasmaUpper) {

        // Multiply U by D
        for (j=1; j<N; j++) {
            alpha = A[LDA*j+j];
            cblas_zscal(j, CBLAS_SADDR(alpha), &A[LDA*j], 1);
        }

    }
}

