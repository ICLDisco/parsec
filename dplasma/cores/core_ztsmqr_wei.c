/**
 *
 * @file core_ztsmqr.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.5.0
 * @author Hatem Ltaief
 * @author Mathieu Faverge
 * @author Jakub Kurzak
 * @author Azzam Haidar
 * @author Dulceneia Becker
 * @date 2010-11-15
 * @precisions normal z -> c d s
 *
 **/
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
 *  CORE_ztsmqr overwrites the general complex M1-by-N1 tile A1 and
 *  M2-by-N2 tile A2 with
 *
 *                        SIDE = 'L'        SIDE = 'R'
 *    TRANS = 'N':         Q * | A1 |     | A1 A2 | * Q
 *                             | A2 |
 *
 *    TRANS = 'C':      Q**H * | A1 |     | A1 A2 | * Q**H
 *                             | A2 |
 *
 *  where Q is a complex unitary matrix defined as the product of k
 *  elementary reflectors
 *
 *    Q = H(1) H(2) . . . H(k)
 *
 *  as returned by CORE_ZTSQRT.
 *
 *******************************************************************************
 *
 * @param[in] side
 *         @arg PlasmaLeft  : apply Q or Q**H from the Left;
 *         @arg PlasmaRight : apply Q or Q**H from the Right.
 *
 * @param[in] trans
 *         @arg PlasmaNoTrans   :  No transpose, apply Q;
 *         @arg PlasmaConjTrans :  ConjTranspose, apply Q**H.
 *
 * @param[in] M1
 *         The number of rows of the tile A1. M1 >= 0.
 *
 * @param[in] N1
 *         The number of columns of the tile A1. N1 >= 0.
 *
 * @param[in] M2
 *         The number of rows of the tile A2. M2 >= 0.
 *         M2 = M1 if side == PlasmaRight.
 *
 * @param[in] N2
 *         The number of columns of the tile A2. N2 >= 0.
 *         N2 = N1 if side == PlasmaLeft.
 *
 * @param[in] K
 *         The number of elementary reflectors whose product defines
 *         the matrix Q.
 *
 * @param[in] IB
 *         The inner-blocking size.  IB >= 0.
 *
 * @param[in,out] A1
 *         On entry, the M1-by-N1 tile A1.
 *         On exit, A1 is overwritten by the application of Q.
 *
 * @param[in] LDA1
 *         The leading dimension of the array A1. LDA1 >= max(1,M1).
 *
 * @param[in,out] A2
 *         On entry, the M2-by-N2 tile A2.
 *         On exit, A2 is overwritten by the application of Q.
 *
 * @param[in] LDA2
 *         The leading dimension of the tile A2. LDA2 >= max(1,M2).
 *
 * @param[in] V
 *         The i-th row must contain the vector which defines the
 *         elementary reflector H(i), for i = 1,2,...,k, as returned by
 *         CORE_ZTSQRT in the first k columns of its array argument V.
 *
 * @param[in] LDV
 *         The leading dimension of the array V. LDV >= max(1,K).
 *
 * @param[in] T
 *         The IB-by-N1 triangular factor T of the block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] LDT
 *         The leading dimension of the array T. LDT >= IB.
 *
 * @param[out] WORK
 *         Workspace array of size
 *             LDWORK-by-N1 if side == PlasmaLeft
 *             LDWORK-by-IB if side == PlasmaRight
 *
 * @param[in] LDWORK
 *         The leading dimension of the array WORK.
 *             LDWORK >= max(1,IB) if side == PlasmaLeft
 *             LDWORK >= max(1,M1) if side == PlasmaRight
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_ztsmqr_wei = PCORE_ztsmqr_wei
#define CORE_ztsmqr_wei PCORE_ztsmqr_wei
#endif
int CORE_ztsmqr_wei(PLASMA_enum side, PLASMA_enum trans,
                int M1, int N1, int M2, int N2, int K, int IB,
                PLASMA_Complex64_t *A1, int LDA1,
                PLASMA_Complex64_t *A2, int LDA2,
                const PLASMA_Complex64_t *V, int LDV,
                const PLASMA_Complex64_t *T, int LDT,
                PLASMA_Complex64_t *WORK, int LDWORK,
                PLASMA_Complex64_t *WORKC, int LDWORKC)
{
    int i, i1, i3;
    int NQ, NW;
    int kb;
    int ic = 0;
    int jc = 0;
    int mi = M1;
    int ni = N1;

    /* Check input arguments */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        coreblas_error(1, "Illegal value of side");
        return -1;
    }

    /* NQ is the order of Q */
    if (side == PlasmaLeft) {
        NQ = M2;
        NW = IB;
    }
    else {
        NQ = N2;
        NW = M1;
    }

    if ((trans != PlasmaNoTrans) && (trans != PlasmaConjTrans)) {
        coreblas_error(2, "Illegal value of trans");
        return -2;
    }
    if (M1 < 0) {
        coreblas_error(3, "Illegal value of M1");
        return -3;
    }
    if (N1 < 0) {
        coreblas_error(4, "Illegal value of N1");
        return -4;
    }
    if ( (M2 < 0) ||
         ( (M2 != M1) && (side == PlasmaRight) ) ){
        coreblas_error(5, "Illegal value of M2");
        return -5;
    }
    if ( (N2 < 0) ||
         ( (N2 != N1) && (side == PlasmaLeft) ) ){
        coreblas_error(6, "Illegal value of N2");
        return -6;
    }
    if ((K < 0) ||
        ( (side == PlasmaLeft)  && (K > M1) ) ||
        ( (side == PlasmaRight) && (K > N1) ) ) {
        coreblas_error(7, "Illegal value of K");
        return -7;
    }
    if (IB < 0) {
        coreblas_error(8, "Illegal value of IB");
        return -8;
    }
    if (LDA1 < max(1,M1)){
        coreblas_error(10, "Illegal value of LDA1");
        return -10;
    }
    if (LDA2 < max(1,M2)){
        coreblas_error(12, "Illegal value of LDA2");
        return -12;
    }
    if (LDV < max(1,NQ)){
        coreblas_error(14, "Illegal value of LDV");
        return -14;
    }
    if (LDT < max(1,IB)){
        coreblas_error(16, "Illegal value of LDT");
        return -16;
    }
    if (LDWORK < max(1,NW)){
        coreblas_error(18, "Illegal value of LDWORK");
        return -18;
    }

    /* Quick return */
    if ((M1 == 0) || (N1 == 0) || (M2 == 0) || (N2 == 0) || (K == 0) || (IB == 0))
        return PLASMA_SUCCESS;

    if (((side == PlasmaLeft)  && (trans != PlasmaNoTrans))
        || ((side == PlasmaRight) && (trans == PlasmaNoTrans))) {
        i1 = 0;
        i3 = IB;
    }
    else {
        i1 = ((K-1) / IB)*IB;
        i3 = -IB;
    }

    for(i = i1; (i > -1) && (i < K); i += i3) {
        kb = min(IB, K-i);

        if (side == PlasmaLeft) {
            /*
             * H or H' is applied to C(i:m,1:n)
             */
            mi = M1 - i;
            ic = i;
        }
        else {
            /*
             * H or H' is applied to C(1:m,i:n)
             */
            ni = N1 - i;
            jc = i;
        }
        /*
         * Apply H or H' (NOTE: CORE_zparfb used to be CORE_ztsrfb)
         */
        CORE_zparfb_wei(
            side, trans, PlasmaForward, PlasmaColumnwise,
            mi, ni, M2, N2, kb, 0,
            &A1[LDA1*jc+ic], LDA1,
            A2, LDA2,
            &V[LDV*i], LDV,
            &T[LDT*i], LDT,
            WORK, LDWORK,
            WORKC, LDWORKC);
    }
    return PLASMA_SUCCESS;
}
