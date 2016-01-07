/**
 *
 * @file core_zhedrk.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.4.5
 * @author Dulceneia Becker
 * @author Mathieu Faverge
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

int CORE_zhedr2(PLASMA_enum uplo, PLASMA_enum trans,
                int N, int K,
                double alpha, PLASMA_Complex64_t *A, int LDA,
                double beta,  PLASMA_Complex64_t *C, int LDC,
                PLASMA_Complex64_t *D, int incD);

int CORE_zhedrk(PLASMA_enum uplo, PLASMA_enum trans,
                int N, int K, int ib,
                double alpha, PLASMA_Complex64_t *A, int LDA,
                double beta,  PLASMA_Complex64_t *C, int LDC,
                PLASMA_Complex64_t *D,    int incD,
                PLASMA_Complex64_t *WORK, int LWORK);

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 * CORE_zhedrk performs one of the matrix-matrix operations
 *
 *       C := alpha*A*D*A' + beta*C  or  C := alpha*A'*D*A + beta*C
 *
 * where alpha and beta are scalars, and A, C and D are matrices, with
 *
 *       A an n by k (ADA') or k x n (A'DA) matrix,
 *       C an n by n matrix and
 *       D an k by k matrix.
 *
 *******************************************************************************
 *
 * @param[in] UPLO
 *          INTEGER
 *          @arg PlasmaLower: Lower triangle of A is stored and scaled.
 *          @arg PlasmaUpper: Upper triangle of A is stored and scaled.
 *
 * @param[in] TRANS
 *         INTEGER
 *         @arg PlasmaNoTrans   :  No transpose, C := alpha*A*A' + beta*C;
 *         @arg PlasmaConjTrans :  Transpose, C := alpha*A'*A + beta*C.
 *
 * @param[in] N
 *         INTEGER
 *         On entry,  N specifies the order of the matrix C.  N must be
 *         at least zero.
 *         Unchanged on exit.
 *
 * @param[in] K
 *         INTEGER
 *         On entry with  TRANS = PlasmaNoTrans,  K  specifies  the number
 *         of  columns   of  the   matrix   A,   and  on   entry   with
 *         TRANS = PlasmaTrans or PlasmaConjTrans,  K  specifies  the  number
 *         of rows of the matrix  A.  K must be at least zero.
 *         Unchanged on exit.
 *
 * @param[in] ALPHA
 *         DOUBLE PRECISION.
 *         On entry, ALPHA specifies the scalar alpha.
 *         Unchanged on exit.
 *
 * @param[in] A
 *         PLASMA_Complex64_t array of DIMENSION ( LDA, ka ), where ka is
 *         k  when  TRANS = PlasmaTrans,  and is  n  otherwise.
 *         Before entry with  TRANS = PlasmaTrans,  the  leading  n by k
 *         part of the array  A  must contain the matrix  A,  otherwise
 *         the leading  k by n  part of the array  A  must contain  the
 *         matrix A.
 *         Unchanged on exit.
 *
 * @param[in] LDA
 *        INTEGER.
 *        On entry, LDA specifies the first dimension of A as declared
 *        in the calling (sub) program. When TRANSA = PlasmaTrans then
 *        LDA must be at least  max( 1, k ), otherwise  LDA must be at
 *        least  max( 1, k ).
 *        Unchanged on exit.
 *
 * @param[in] BETA
 *       DOUBLE PRECISION.
 *       On entry,  BETA  specifies the scalar  beta.
 *       Unchanged on exit.
 *
 * @param[in] C
 *       PLASMA_Complex64_t array of DIMENSION ( LDC, n ).
 *
 *       DB_TEMP FOR NOW, BOTH UPPER AND LOWER PARTS OF C MUST BE STORED IN C.
 *
 *       DB_TEMP AFTER CORRECTION:
 *
 *       Before entry  with  UPLO = PlasmaUpper,  the leading  n by n
 *       upper triangular part of the array C must contain the upper
 *       triangular part  of the  symmetric matrix  and the strictly
 *       lower triangular part of C is not referenced.  On exit, the
 *       upper triangular part of the array  C is overwritten by the
 *       upper triangular part of the updated matrix.
 *       Before entry  with  UPLO = PlasmaLower,  the leading  n by n
 *       lower triangular part of the array C must contain the lower
 *       triangular part  of the  symmetric matrix  and the strictly
 *       upper triangular part of C is not referenced.  On exit, the
 *       lower triangular part of the array  C is overwritten by the
 *       lower triangular part of the updated matrix.
 *
 * @param[in] LDC
 *       INTEGER
 *       On entry, LDC specifies the first dimension of C as declared
 *       in  the  calling  (sub)  program.   LDC  must  be  at  least
 *       max( 1, m ).
 *       Unchanged on exit.
 *
 * @param[workspace] WORK
 *       PLASMA_Complex64_t array, dimension (MAX(1,LWORK))
 *
 * @param[in] LWORK
 *       INTEGER
 *       The length of WORK.  LWORK >= max(1, N*K)
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/

#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_zhedr2 = PCORE_zhedr2
#define CORE_zhedr2 PCORE_zhedr2
#endif
int CORE_zhedr2(PLASMA_enum uplo, PLASMA_enum trans,
                int N, int K,
                double alpha, PLASMA_Complex64_t *A, int LDA,
                double beta,  PLASMA_Complex64_t *C, int LDC,
                PLASMA_Complex64_t *D, int incD)
{
    int i, j, k, Am;
    PLASMA_Complex64_t tmp;
    PLASMA_Complex64_t *Aik, *Dkk, *Akj, *Cij;

    Am = (trans == PlasmaNoTrans ) ? N : K;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        coreblas_error(1, "Illegal value of uplo");
        return -1;
    }
    if ((trans != PlasmaNoTrans) && (trans != PlasmaConjTrans)) {
        coreblas_error(2, "Illegal value of trans");
        return -2;
    }
    if (N < 0) {
        coreblas_error(3, "Illegal value of N");
        return -3;
    }
    if (K < 0) {
        coreblas_error(4, "Illegal value of K");
        return -4;
    }
    if ((LDA < max(1,Am)) && (Am > 0)) {
        coreblas_error(7, "Illegal value of LDA");
        return -7;
    }
    if ((LDC < max(1,N)) && (N > 0)) {
        coreblas_error(10, "Illegal value of LDC");
        return -10;
    }
    if ( incD < 0 ) {
        coreblas_error(12, "Illegal value of incD");
        return -12;
    }

    /* Quick return */
    if (N == 0 || K == 0 ||
        ((alpha == 0.0 || K == 0) && beta == 1.0) ) {
        return PLASMA_SUCCESS;
    }

    if ( uplo == PlasmaLower )
    {
        /*
         * PlasmaLower / PlasmaNoTrans
         */
        if ( trans == PlasmaNoTrans )
        {
            Cij = C;

            for(j=0; j<N; j++)
            {
                for(i=j; i<N; i++, Cij++)
                {
                    tmp = 0.0;
                    Aik = A+i;
                    Dkk = D;
                    Akj = A+LDA*j;
                    for(k=0; k<K; k++, Aik+=LDA, Dkk++, Akj++ )
                    {
                        tmp += (*Aik) * (*Dkk) * conj( *Akj );
                    }
                    *Cij = beta * (*Cij) + alpha * tmp;
                }
                Cij += LDC-N+j;
            }
        }
        /*
         * PlasmaLower / PlasmaConjTrans
         */
        else
        {
            for(j=0; j<N; j++)
            {
                for(i=j; i<N; i++)
                {
                    tmp = 0.0;
                    for(k=0; k<K; k++)
                    {
                        tmp += conj(A[LDA*i+k]) * D[incD*k] * A[LDA*k+j];
                    }
                    C[LDC*j+i] = beta * C[LDC*j+i] + alpha * tmp;
                }
            }
        }
    }
    else
    {
        /*
         * PlasmaUpper / PlasmaNoTrans
         */
        if ( trans == PlasmaNoTrans )
        {
            for(j=0; j<N; j++)
            {
                for(i=0; i<=j; i++)
                {
                    tmp = 0.0;
                    for(k=0; k<K; k++)
                    {
                        tmp += A[LDA*k+i] * D[incD*k] * conj( A[LDA*j+k] );
                    }
                    C[LDC*j+i] = beta * C[LDC*j+i] + alpha * tmp;
                }
            }
        }
        /*
         * PlasmaUpper / PlasmaConjTrans
         */
        else
        {
            for(j=0; j<N; j++)
            {
                for(i=j; i<N; i++)
                {
                    tmp = 0.0;
                    for(k=0; k<K; k++)
                    {
                        tmp += conj(A[LDA*i+k]) * D[incD*k] * A[LDA*k+j];
                    }
                    C[LDC*j+i] = beta * C[LDC*j+i] + alpha * tmp;
                }
            }
        }
    }
    return 0;
}

#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_zhedrk = PCORE_zhedrk
#define CORE_zhedrk PCORE_zhedrk
#endif
int CORE_zhedrk(PLASMA_enum uplo, PLASMA_enum trans,
                int N, int K, int ib,
                double alpha, PLASMA_Complex64_t *A, int LDA,
                double beta,  PLASMA_Complex64_t *C, int LDC,
                PLASMA_Complex64_t *D,    int incD,
                PLASMA_Complex64_t *WORK, int LWORK)
{
    int i, j, ii, sb, Am;
    PLASMA_Complex64_t *wD, *AD, *wDC;
    PLASMA_Complex64_t *X, *Y;
    PLASMA_Complex64_t zzero  = (PLASMA_Complex64_t)0.;
    PLASMA_Complex64_t zalpha = alpha;
    PLASMA_Complex64_t zbeta  = beta;

    Am = (trans == PlasmaNoTrans ) ? N : K;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        coreblas_error(1, "Illegal value of uplo");
        return -1;
    }
    if ((trans != PlasmaNoTrans) && (trans != PlasmaConjTrans)) {
        coreblas_error(2, "Illegal value of trans");
        return -2;
    }
    if (N < 0) {
        coreblas_error(3, "Illegal value of N");
        return -3;
    }
    if (K < 0) {
        coreblas_error(4, "Illegal value of K");
        return -4;
    }
    if (ib < 0) {
        coreblas_error(5, "Illegal value of ib");
        return -5;
    }
    if ((LDA < max(1,Am)) && (Am > 0)) {
        coreblas_error(8, "Illegal value of LDA");
        return -8;
    }
    if ((LDC < max(1,N)) && (N > 0)) {
        coreblas_error(11, "Illegal value of LDC");
        return -11;
    }
    if ( incD < 0 ) {
        coreblas_error(13, "Illegal value of incD");
        return -13;
    }

    if ( LWORK < ((N+1)*K + ib*ib) ) {
        coreblas_error(15, "Illegal value of LWORK");
        return -15;
    }

    /* Quick return */
    if (N == 0 || K == 0 ||
        ((alpha == 0.0 || K == 0) && beta == 1.0) ) {
        return PLASMA_SUCCESS;
    }

    if ( incD == 1 ) {
        wD = D;
    } else {
        wD = WORK;
        cblas_zcopy(K, D, incD, wD, 1);
    }
    AD  = WORK + K;
    wDC = AD + N*K;

    /* Compute (A * D) */
    if ( trans == PlasmaNoTrans )
    {
        /* AD = A * D */
        for (j=0; j<K; j++, wD++) {
            cblas_zcopy(N, A + LDA*j, 1,       AD + N*j, 1);
            cblas_zscal(N, CBLAS_SADDR((*wD)), AD + N*j, 1);
        }
    }
    else {
        /* AD = (D * A)' */
        for (j=0; j<K; j++, wD++) {
            cblas_zcopy(N, A + j,      LDA,    AD + N*j, 1);
            cblas_zscal(N, CBLAS_SADDR((*wD)), AD + N*j, 1);
        }
    }

    if ( uplo == PlasmaLower )
    {
        if ( trans == PlasmaNoTrans )
        {
            for( ii=0; ii<N; ii+=ib )
            {
                sb = min(N-ii, ib);

                /* W = alpha * (A * D) * A' */
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                            sb, sb, K,
                            CBLAS_SADDR(zalpha), AD+ii, N,
                                                 A +ii, LDA,
                            CBLAS_SADDR(zzero),  wDC,   sb);

                /* Ckk = beta * Ckk + W = beta * Ckk +  alpha * (A * D) * A' */
                {
                    X = wDC;
                    Y = C + ii * LDC + ii;
                    for (j=0; j<sb; j++) {
                        for(i=j; i<sb; i++, Y++, X++) {
                            *Y = beta * (*Y) + (*X);
                        }
                        Y += LDC-sb+j+1;
                        X += j+1;
                    }
                }

                /* C = alpha * (A*D) * A' + beta * C */
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                            N-ii-sb, sb, K,
                            CBLAS_SADDR(zalpha), AD+          ii+sb, N,
                                                 A + ii,             LDA,
                            CBLAS_SADDR(zbeta),  C + LDC*ii + ii+sb, LDC);
            }
        }
        /*
         * PlasmaLower / PlasmaConjTrans
         */
        else {
            for( ii=0; ii<N; ii+=ib )
            {
                sb = min(N-ii, ib);

                /* W = alpha * A' * (D * A) */
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasTrans,
                            sb, sb, K,
                            CBLAS_SADDR(zalpha), A +LDA*ii, LDA,
                                                 AD+ii,     N,
                            CBLAS_SADDR(zzero),  wDC,       sb);

                /* Ckk = beta * Ckk + W = beta * Ckk +  alpha * A' * (D * A) */
                {
                    X = wDC;
                    Y = C + ii * LDC + ii;
                    for (j=0; j<sb; j++) {
                        for(i=j; i<sb; i++, Y++, X++) {
                            *Y = beta * (*Y) + (*X);
                        }
                        Y += LDC-sb+j+1;
                        X += j+1;
                    }
                }

                /* C = alpha * A' * (D * A) + beta * C */
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasTrans,
                            N-ii-sb, sb, K,
                            CBLAS_SADDR(zalpha), A + LDA*ii,       LDA,
                                                 AD+        ii+sb, N,
                            CBLAS_SADDR(zbeta),  C + LDC*ii+ii+sb, LDC);
            }
        }
    }
    else
    {
        /*
         * PlasmaUpper / PlasmaNoTrans
         */
        if ( trans == PlasmaNoTrans )
        {
            for( ii=0; ii<N; ii+=ib )
            {
                sb = min(N-ii, ib);

                /* W = alpha * (A * D) * A' */
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                            sb, sb, K,
                            CBLAS_SADDR(zalpha), AD+ii, N,
                                                 A +ii, LDA,
                            CBLAS_SADDR(zzero),  wDC,   sb);

                /* Ckk = beta * Ckk + W = beta * Ckk +  alpha * (A * D) * A' */
                {
                    X = wDC;
                    Y = C + LDC*ii + ii;
                    for (j=0; j<sb; j++) {
                        int mm = min( j+1, sb );
                        for(i=0; i<mm; i++, Y++, X++) {
                            *Y = zbeta * (*Y) + (*X);
                        }
                        Y += LDC-mm;
                        X += sb -mm;
                    }
                }

                /* C = alpha * (A*D) * A' + beta * C */
                cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                            sb, N-ii-sb, K,
                            CBLAS_SADDR(zalpha), AD+             ii, N,
                                                 A +      ii+sb,     LDA,
                            CBLAS_SADDR(zbeta),  C + LDC*(ii+sb)+ii, LDC);
            }
        }
        /*
         * PlasmaUpper / PlasmaConjTrans
         */
        else {
            for( ii=0; ii<N; ii+=ib )
            {
                sb = min(N-ii, ib);

                /* W = alpha * A' * (D * A) */
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasTrans,
                            sb, sb, K,
                            CBLAS_SADDR(zalpha), A +LDA*ii, LDA,
                                                 AD+ii,     N,
                            CBLAS_SADDR(zzero),  wDC,       sb);

                /* Ckk = beta * Ckk + W = beta * Ckk +  alpha * A' * (D * A) */
                {
                    X = wDC;
                    Y = C + LDC*ii + ii;
                    for (j=0; j<sb; j++) {
                        int mm = min( j+1, sb );
                        for(i=0; i<mm; i++, Y++, X++) {
                            *Y = zbeta * (*Y) + (*X);
                        }
                        Y += LDC-mm;
                        X += sb -mm;
                    }
                }

                /* C = alpha * A' * (D * A) + beta * C */
                cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                            sb, N-ii-sb, K,
                            CBLAS_SADDR(zalpha), A +LDA*(ii+sb),    LDA,
                                                 AD+            ii, N,
                            CBLAS_SADDR(zbeta),  C +LDC*(ii+sb)+ii, LDC);
            }
        }
    }

    return 0;
}
