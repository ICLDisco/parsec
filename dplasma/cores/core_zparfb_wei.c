/**
 *
 * @file core_zparfb.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.5.0
 * @author Dulceneia Becker
 * @date 2011-06-14
 * @precisions normal z -> c d s
 *
 **/
#include <lapacke.h>
#include "dague_config.h"
#include "data_dist/matrix/precision.h"
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

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  CORE_zparfb applies a complex upper triangular block reflector H
 *  or its transpose H' to a complex rectangular matrix formed by
 *  coupling two tiles A1 and A2. Matrix V is:
 *
 *          COLUMNWISE                    ROWWISE
 *
 *         |     K     |                 |      N2-L     |   L  |
 *      __ _____________ __           __ _________________        __
 *         |    |      |                 |               | \
 *         |    |      |                 |               |   \    L
 *    M2-L |    |      |              K  |_______________|_____\  __
 *         |    |      | M2              |                      |
 *      __ |____|      |                 |                      | K-L
 *         \    |      |              __ |______________________| __
 *       L   \  |      |
 *      __     \|______| __              |          N2          |
 *
 *         | L |  K-L  |
 *
 *******************************************************************************
 *
 * @param[in] side
 *         @arg PlasmaLeft  : apply Q or Q**H from the Left;
 *         @arg PlasmaRight : apply Q or Q**H from the Right.
 *
 * @param[in] trans
 *         @arg PlasmaNoTrans   : No transpose, apply Q;
 *         @arg PlasmaConjTrans : ConjTranspose, apply Q**H.
 *
 * @param[in] direct
 *         Indicates how H is formed from a product of elementary
 *         reflectors
 *         @arg PlasmaForward  : H = H(1) H(2) . . . H(k) (Forward)
 *         @arg PlasmaBackward : H = H(k) . . . H(2) H(1) (Backward)
 *
 * @param[in] storev
 *         Indicates how the vectors which define the elementary
 *         reflectors are stored:
 *         @arg PlasmaColumnwise
 *         @arg PlasmaRowwise
 *
 * @param[in] M1
 *         The number of columns of the tile A1. M1 >= 0.
 *
 * @param[in] N1
 *         The number of rows of the tile A1. N1 >= 0.
 *
 * @param[in] M2
 *         The number of columns of the tile A2. M2 >= 0.
 *
 * @param[in] N2
 *         The number of rows of the tile A2. N2 >= 0.
 *
 * @param[in] K
 *         The order of the matrix T (= the number of elementary
 *         reflectors whose product defines the block reflector).
 *
 * @param[in] L
 *         The size of the triangular part of V
 *
 * @param[in,out] A1
 *         On entry, the M1-by-N1 tile A1.
 *         On exit, A1 is overwritten by the application of Q.
 *
 * @param[in] LDA1
 *         The leading dimension of the array A1. LDA1 >= max(1,N1).
 *
 * @param[in,out] A2
 *         On entry, the M2-by-N2 tile A2.
 *         On exit, A2 is overwritten by the application of Q.
 *
 * @param[in] LDA2
 *         The leading dimension of the tile A2. LDA2 >= max(1,N2).
 *
 * @param[in] V
 *         (LDV,K) if STOREV = 'C'
 *         (LDV,M2) if STOREV = 'R' and SIDE = 'L'
 *         (LDV,N2) if STOREV = 'R' and SIDE = 'R'
 *         Matrix V.
 *
 * @param[in] LDV
 *         The leading dimension of the array V.
 *         If STOREV = 'C' and SIDE = 'L', LDV >= max(1,M2);
 *         if STOREV = 'C' and SIDE = 'R', LDV >= max(1,N2);
 *         if STOREV = 'R', LDV >= K.
 *
 * @param[out] T
 *         The triangular K-by-K matrix T in the representation of the
 *         block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] LDT
 *         The leading dimension of the array T. LDT >= K.
 *
 * @param[in,out] WORK
 *
 * @param[in] LDWORK
 *         The dimension of the array WORK.
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
/* This kernel is never traced so return type on previous line for convert2eztrace.pl script */
int
CORE_zparfb_wei(PLASMA_enum side, PLASMA_enum trans, PLASMA_enum direct, PLASMA_enum storev,
            int M1, int N1, int M2, int N2, int K, int L,
                  PLASMA_Complex64_t *A1, int LDA1,
                  PLASMA_Complex64_t *A2, int LDA2,
            const PLASMA_Complex64_t *V, int LDV,
            const PLASMA_Complex64_t *T, int LDT,
                  PLASMA_Complex64_t *WORK, int LDWORK,
                  PLASMA_Complex64_t *WORKC, int LDWORKC)
{
    static PLASMA_Complex64_t zone  =  1.0;
    static PLASMA_Complex64_t mzone = -1.0;

    int j;

    /* Check input arguments */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        coreblas_error(1, "Illegal value of side");
        return -1;
    }
    if ((trans != PlasmaNoTrans) && (trans != PlasmaConjTrans)) {
        coreblas_error(2, "Illegal value of trans");
        return -2;
    }
    if ((direct != PlasmaForward) && (direct != PlasmaBackward)) {
        coreblas_error(3, "Illegal value of direct");
        return -3;
    }
    if ((storev != PlasmaColumnwise) && (storev != PlasmaRowwise)) {
        coreblas_error(4, "Illegal value of storev");
        return -4;
    }
    if (M1 < 0) {
        coreblas_error(5, "Illegal value of M1");
        return -5;
    }
    if (N1 < 0) {
        coreblas_error(6, "Illegal value of N1");
        return -6;
    }
    if ((M2 < 0) ||
        ( (side == PlasmaRight) && (M1 != M2) ) ) {
        coreblas_error(7, "Illegal value of M2");
        return -7;
    }
    if ((N2 < 0) ||
        ( (side == PlasmaLeft) && (N1 != N2) ) ) {
        coreblas_error(8, "Illegal value of N2");
        return -8;
    }
    if (K < 0) {
        coreblas_error(9, "Illegal value of K");
        return -9;
    }

    /* Quick return */
    if ((M1 == 0) || (N1 == 0) || (M2 == 0) || (N2 == 0) || (K == 0))
        return PLASMA_SUCCESS;

    if (direct == PlasmaForward) {

        if (side == PlasmaLeft) {

            /*
             * Column or Rowwise / Forward / Left
             * ----------------------------------
             *
             * Form  H * A  or  H' * A  where  A = ( A1 )
             *                                     ( A2 )
             */

            /* W = A1 + op(V) * A2 */
            CORE_zpamm(
                    PlasmaW, PlasmaLeft, storev,
                    K, N1, M2, L,
                    A1, LDA1,
                    A2, LDA2,
                    V, LDV,
                    WORK, LDWORK);

            /* W = op(T) * W */
            if (WORKC == NULL) {
                cblas_ztrmm(
                    CblasColMajor, CblasLeft, CblasUpper,
                    (CBLAS_TRANSPOSE)trans, CblasNonUnit, K, N2,
                    CBLAS_SADDR(zone), T, LDT, WORK, LDWORK);
                WORKC = WORK;
                LDWORKC = LDWORK;
            } else {
                cblas_zgemm(
                CblasColMajor, 	
                (CBLAS_TRANSPOSE)trans, CblasNoTrans,
                K, N2, K,
                CBLAS_SADDR(zone),
                T, LDT,
                WORK, LDWORK,
                0.0,
                WORKC, LDWORKC);
            }

            /* A1 = A1 - W */
            for(j = 0; j < N1; j++) {
                cblas_zaxpy(
                        K, CBLAS_SADDR(mzone),
                        &WORKC[LDWORKC*j], 1,
                        &A1[LDA1*j], 1);
            }

            /* A2 = A2 - op(V) * W  */
            /* W also changes: W = V * W, A2 = A2 - W */
            CORE_zpamm(
                    PlasmaA2, PlasmaLeft, storev,
                    M2, N2, K, L,
                    A1, LDA1,
                    A2, LDA2,
                    V, LDV,
                    WORKC, LDWORKC);
        }
        else {
            /*
             * Column or Rowwise / Forward / Right
             * -----------------------------------
             *
             * Form  H * A  or  H' * A  where A  = ( A1 A2 )
             *
             */

            /* W = A1 + A2 * op(V) */
            CORE_zpamm(
                    PlasmaW, PlasmaRight, storev,
                    M1, K, N2, L,
                    A1, LDA1,
                    A2, LDA2,
                    V, LDV,
                    WORK, LDWORK);

            /* W = W * op(T) */
            cblas_ztrmm(
                CblasColMajor, CblasRight, CblasUpper,
                (CBLAS_TRANSPOSE)trans, CblasNonUnit, M2, K,
                CBLAS_SADDR(zone), T, LDT, WORK, LDWORK);

            /* A1 = A1 - W */
            for(j = 0; j < K; j++) {
                cblas_zaxpy(
                        M1, CBLAS_SADDR(mzone),
                        &WORK[LDWORK*j], 1,
                        &A1[LDA1*j], 1);
            }

            /* A2 = A2 - W * op(V) */
            /* W also changes: W = W * V', A2 = A2 - W */
            CORE_zpamm(
                    PlasmaA2, PlasmaRight, storev,
                    M2, N2, K, L,
                    A1, LDA1,
                    A2, LDA2,
                    V, LDV,
                    WORK, LDWORK);
        }
    }
    else {
        coreblas_error(3, "Not implemented (Backward / Left or Right)");
        return PLASMA_ERR_NOT_SUPPORTED;
    }

    return PLASMA_SUCCESS;
}
