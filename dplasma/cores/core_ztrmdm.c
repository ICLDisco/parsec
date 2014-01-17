/**
 *
 * @file core_ztrmdm.c
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

#include <cblas.h>
#include <core_blas.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

int CORE_ztrmdm(int uplo, int N, PLASMA_Complex64_t *A, int LDA);

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 * CORE_ztrmdm scales the strictly upper or strictly lower triangular part of a
 * square matrix by the inverse of a diagonal matrix, ie performs either
 *
 *    A := L / D  or  A := U / D (only for triangular part above or below diagonal)
 *
 * where:
 *
 *    L is a strictly lower triangular matrix stored as the strictly lower triangle in A
 *    U is a strictly upper triangular matrix stored as the strictly upper triangle in A
 *    D is a diagonal matrix stored as the diagonal in A
 *
 * The diagonal elements of A are not changed.
 *
*******************************************************************************
 *
 * @param[in] UPLO
 *         INTEGER
 *         @arg PlasmaLower: Lower triangle of A is stored and scaled.
 *         @arg PlasmaUpper: Upper triangle of A is stored and scaled.
 *
 * @param[in] N
 *         INTEGER
 *         The number of rows and columns of A.  N >= 0.
 *
 * @param[in,out] A
 *         PLASMA_Complex64_t array, dimension (LDA,N)
 *
 *         On entry, the triangular matrix A. If UPLO = 'U', the leading
 *         N-by-N upper triangular part of A contains the upper
 *         triangular part of the matrix A, and the strictly lower
 *         triangular part of A is not referenced.  If UPLO = 'L', the
 *         leading N-by-N lower triangular part of A contains the lower
 *         triangular part of the matrix A, and the strictly upper
 *         triangular part of A is not referenced.
 *
 *         On exit, the strictly lower or upper triangular part of A
 *         scaled by the diagonal elements of A.
 *
 * @param[in] LDA
 *         INTEGER
 *         The leading dimension of the array A.  LDA >= max(1,N).
 *
*******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_ztrmdm = PCORE_ztrmdm
#define CORE_ztrmdm PCORE_ztrmdm
#endif
int CORE_ztrmdm(int uplo, int N, PLASMA_Complex64_t *A, int LDA)
{
    static PLASMA_Complex64_t zone = 1.0;

    PLASMA_Complex64_t alpha;
    int j;

    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        coreblas_error(1, "Illegal value of UPLO");
        return -1;
    }
    if (N < 0) {
        coreblas_error(2, "Illegal value of N");
        return -2;
    }
    if (LDA < max(1, N)) {
        coreblas_error(1, "Illegal value of LDA");
        return -4;
    }

    /* Quick return */
    if (max(N, 0) == 0)
        return PLASMA_SUCCESS;

    /**/

    if (uplo==PlasmaLower) {
        for (j=0; j<(N-1); j++) {
                alpha = zone / A[LDA*j+j];
                cblas_zscal(N-j-1, CBLAS_SADDR(alpha), &A[LDA*j+j+1], 1);
        }

    } else if (uplo==PlasmaUpper) {
        for (j=1; j<N; j++) {
                alpha = zone / A[LDA*j+j];
                cblas_zscal(j, CBLAS_SADDR(alpha), &A[LDA*j], 1);
        }

    }

    return PLASMA_SUCCESS;

}
