/**
 *
 * @file core_zgemdm.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.4.5
 * @author Dulceneia Becker
 * @author Mathieu Faverge
 * @date 2011-1-18
 */
/*
 * @precisions normal z -> c d s
 */
#include <lapacke.h>
#include "dplasma_cores.h"
#include "dplasma_zcores.h"

#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */
#if defined(PARSEC_HAVE_STDARG_H)
#include <stdarg.h>
#endif  /* defined(PARSEC_HAVE_STDARG_H) */
#include <stdio.h>
#ifdef PARSEC_HAVE_LIMITS_H
#include <limits.h>
#endif

#include <cblas.h>
#include <core_blas.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

int CORE_zgemdm(int transA, int transB,
                int M, int N, int K,
                PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA,
                PLASMA_Complex64_t *B, int LDB,
                PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int LDC,
                PLASMA_Complex64_t *D, int incD,
                PLASMA_Complex64_t *WORK, int LWORK);

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 * CORE_zgemdm performs one of the matrix-matrix operations
 *
 *       C := alpha*op( A )*D*op( B ) + beta*C,
 *
 * where op( X ) is one of
 *
 *       op( X ) = X   or   op( X ) = X',
 *
 * alpha and beta are scalars, and A, B, C and D are matrices, with
 *
 *       op( A ) an m by k matrix,
 *       op( B ) an k by n matrix,
 *       C an m by n matrix and
 *       D an k by k matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *         INTEGER
 *         @arg PlasmaNoTrans   :  No transpose, op( A ) = A;
 *         @arg PlasmaConjTrans :  Transpose, op( A ) = A'.
 *
 * @param[in] transB
 *         INTEGER
 *         @arg PlasmaNoTrans   :  No transpose, op( B ) = B;
 *         @arg PlasmaConjTrans :  Transpose, op( B ) = B'.
 *
 * @param[in] M
 *         INTEGER
 *         The number of rows  of the  matrix op( A ) and of the
 *         matrix C.  M  must  be at least  zero.
 *
 * @param[in] N
 *         INTEGER
 *         The number of columns of the matrix  op( B ) and the
 *         number of columns of the matrix C. N must be at least zero.
 *
 * @param[in] K
 *         INTEGER
 *         The number of columns of the matrix op( A ), the number of
 *         rows of the matrix op( B ), and the number of rows and columns
 *         of matrix D. K must be at least  zero.
 *
 * @param[in] alpha
 *         PLASMA_Complex64_t.
 *         On entry, ALPHA specifies the scalar alpha.
 *         Unchanged on exit.
 *
 * @param[in] A
 *         PLASMA_Complex64_t array of DIMENSION ( LDA, ka ), where ka is
 *         k  when  TRANSA = PlasmaTrans, and is  m  otherwise.
 *         Before entry with  TRANSA = PlasmaTrans,  the leading  m by k
 *         part of the array  A  must contain the matrix  A,  otherwise
 *         the leading  k by m  part of the array  A  must contain  the
 *         matrix A.
 *         Unchanged on exit.
 *
 * @param[in] LDA
 *        INTEGER.
 *        On entry, LDA specifies the first dimension of A as declared
 *        in the calling (sub) program. When TRANSA = PlasmaTrans then
 *        LDA must be at least  max( 1, m ), otherwise  LDA must be at
 *        least  max( 1, k ).
 *        Unchanged on exit.
 *
 * @param[in] B
 *        PLASMA_Complex64_t array of DIMENSION ( LDB, kb ), where kb is
 *        n  when TRANSB = PlasmaTrans, and is k otherwise.
 *        Before entry with TRANSB = PlasmaTrans, the leading  k by n
 *        part of the array  B  must contain the matrix B, otherwise
 *        the leading n by k part of the array B must contain  the
 *        matrix B.
 *        Unchanged on exit.
 *
 * @param[in] LDB
 *       INTEGER.
 *       On entry, LDB specifies the first dimension of B as declared
 *       in the calling (sub) program. When  TRANSB = PlasmaTrans then
 *       LDB must be at least  max( 1, k ), otherwise  LDB must be at
 *       least  max( 1, n ).
 *       Unchanged on exit.
 *
 * @param[in] beta
 *       PLASMA_Complex64_t.
 *       On entry,  BETA  specifies the scalar  beta.  When  BETA  is
 *       supplied as zero then C need not be set on input.
 *       Unchanged on exit.
 *
 * @param[in] C
 *       PLASMA_Complex64_t array of DIMENSION ( LDC, n ).
 *       Before entry, the leading  m by n  part of the array  C must
 *       contain the matrix  C,  except when  beta  is zero, in which
 *       case C need not be set on entry.
 *       On exit, the array  C  is overwritten by the  m by n  matrix
 *       ( alpha*op( A )*D*op( B ) + beta*C ).
 *
 * @param[in] LDC
 *       INTEGER
 *       On entry, LDC specifies the first dimension of C as declared
 *       in  the  calling  (sub)  program.   LDC  must  be  at  least
 *       max( 1, m ).
 *       Unchanged on exit.
 *
 * @param[in] D
 *        PLASMA_Complex64_t array of DIMENSION ( LDD, k ).
 *        Before entry, the leading  k by k part of the array  D
 *        must contain the matrix D.
 *        Unchanged on exit.
 *
 * @param[in] incD
 *       INTEGER.
 *       On entry, incD specifies the first dimension of D as declared
 *       in  the  calling  (sub)  program.   incD  must  be  at  least
 *       max( 1, k ).
 *       Unchanged on exit.
 *
 * @param[inout] WORK
 *       PLASMA_Complex64_t array, dimension (MAX(1,LWORK))
 *
 * @param[in] LWORK
 *       INTEGER
 *       The length of WORK.
 *       On entry, if TRANSA = PlasmaTrans and TRANSB = PlasmaTrans then
 *       LWORK >= max(1, K*N). Otherwise LWORK >= max(1, M*K).
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
#if defined(PLASMA_PARSEC_HAVE_WEAK)
#pragma weak CORE_zgemdm = PCORE_zgemdm
#define CORE_zgemdm PCORE_zgemdm
#endif
int CORE_zgemdm(int transA, int transB,
                int M, int N, int K,
                PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA,
                PLASMA_Complex64_t *B, int LDB,
                PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int LDC,
                PLASMA_Complex64_t *D, int incD,
                PLASMA_Complex64_t *WORK, int LWORK)
{
    int j, Am, Bm;
    PLASMA_Complex64_t delta;
    PLASMA_Complex64_t *wD, *w;
    
    Am = (transA == PlasmaNoTrans ) ? M : K;
    Bm = (transB == PlasmaNoTrans ) ? K : N;

    /* Check input arguments */
    if ((transA != PlasmaNoTrans) && (transA != PlasmaTrans) && (transA != PlasmaConjTrans)) {
        coreblas_error(1, "Illegal value of transA");
        return -1;
    }
    if ((transB != PlasmaNoTrans) && (transB != PlasmaTrans) && (transB != PlasmaConjTrans)) {
        coreblas_error(2, "Illegal value of transB");
        return -2;
    }
    if (M < 0) {
        coreblas_error(3, "Illegal value of M");
        return -3;
    }
    if (N < 0) {
        coreblas_error(4, "Illegal value of N");
        return -4;
    }
    if (K < 0) {
        coreblas_error(5, "Illegal value of K");
        return -5;
    }
    if ((LDA < max(1,Am)) && (Am > 0)) {
        coreblas_error(8, "Illegal value of LDA");
        return -8;
    }
    if ((LDB < max(1,Bm)) && (Bm > 0)) {
        coreblas_error(10, "Illegal value of LDB");
        return -10;
    }
    if ((LDC < max(1,M)) && (M > 0)) {
        coreblas_error(13, "Illegal value of LDC");
        return -13;
    }
    if ( incD < 0 ) {
        coreblas_error(15, "Illegal value of incD");
        return -15;
    }
    if ( ( ( transA == PlasmaNoTrans ) && ( LWORK < (M+1)*K) ) ||
         ( ( transA != PlasmaNoTrans ) && ( LWORK < (N+1)*K) ) ){
        coreblas_error(17, "Illegal value of LWORK");
        return -17;
    }
    
    /* Quick return */
    if (M == 0 || N == 0 || 
        ((alpha == 0.0 || K == 0) && beta == 1.0) ) {
        return PLASMA_SUCCESS;
    }

    if ( incD == 1 ) {
        wD = D;
    } else {
        wD = WORK;
        cblas_zcopy(K, D, incD, wD, 1);
    }
    w = WORK + K;

    /*
     * transA == PlasmaNoTrans
     */ 
    if ( transA == PlasmaNoTrans ) 
    {
        /* WORK = A * D */
      for (j=0; j<K; j++, wD++) {
            delta = *wD;
            cblas_zcopy(M, &A[LDA*j], 1,       &w[M*j], 1);
            cblas_zscal(M, CBLAS_SADDR(delta), &w[M*j], 1);
        }
        
        /* C = alpha * WORK * op(B) + beta * C */
        cblas_zgemm(CblasColMajor, (CBLAS_TRANSPOSE)PlasmaNoTrans, (CBLAS_TRANSPOSE)transB, 
                    M, N, K, 
                    CBLAS_SADDR(alpha), w, M, 
                                        B, LDB, 
                    CBLAS_SADDR(beta),  C, LDC);
    }        
    else 
    {
        if ( transB == PlasmaNoTrans ) /* Worst case*/
        {
            /* WORK = (D * B)' */
          for (j=0; j<K; j++, wD++) {
                delta = *wD;
                cblas_zcopy(N, &B[j],     LDB,     &w[N*j], 1);
                cblas_zscal(N, CBLAS_SADDR(delta), &w[N*j], 1);
            }

            /* C = alpha * op(A) * WORK' + beta * C */
            cblas_zgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)PlasmaTrans,
                        M, N, K, 
                        CBLAS_SADDR(alpha), A, LDA, 
                                            w, N,
                        CBLAS_SADDR(beta),  C, LDC);
        }
        else 
        {
#ifdef COMPLEX
            if ( transB == PlasmaConjTrans )
            {
                /* WORK = D * B' */
              for (j=0; j<K; j++, wD++) {
                    delta = *wD;
                    cblas_zcopy(N, &B[LDB*j], 1,       &w[N*j], 1);
                    LAPACKE_zlacgv_work(N,             &w[N*j], 1);
                    cblas_zscal(N, CBLAS_SADDR(delta), &w[N*j], 1);
                }
            }
            else 
#endif
            {
                /* WORK = D * B' */
              for (j=0; j<K; j++, wD++) {
                    delta = *wD;
                    cblas_zcopy(N, &B[LDB*j], 1,       &w[N*j], 1);
                    cblas_zscal(N, CBLAS_SADDR(delta), &w[N*j], 1);
                }
            }
        
            /* C = alpha * op(A) * WORK + beta * C */
            cblas_zgemm(CblasColMajor, (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)PlasmaNoTrans,
                        M, N, K, 
                        CBLAS_SADDR(alpha), A, LDA, 
                                            w, N, 
                        CBLAS_SADDR(beta),  C, LDC);
        }
    }
    return PLASMA_SUCCESS;
}

