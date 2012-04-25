/**
 *
 * @file core_zhetrf_nopiv.c
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
#include <math.h>
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


#define COMPLEX
#undef REAL


extern int CORE_zhedrk(PLASMA_enum uplo, PLASMA_enum trans,
                int N, int K, int ib,
                double alpha, PLASMA_Complex64_t *A, int LDA,
                double beta,  PLASMA_Complex64_t *C, int LDC,
                PLASMA_Complex64_t *D,    int incD,
                PLASMA_Complex64_t *WORK, int LWORK);

int CORE_zhetf1_nopiv(PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA);

int CORE_zhetf3_nopiv(PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA);

void CORE_zhetrf_nopiv(int uplo, int N, int ib,
                         PLASMA_Complex64_t *A, int LDA,
                         PLASMA_Complex64_t *WORK, int LDWORK,
                         int *INFO);

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 **/
#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_zhetrf_nopiv = PCORE_zhetrf_nopiv
#define CORE_zhetrf_nopiv PCORE_zhetrf_nopiv
#endif
void CORE_zhetrf_nopiv(int uplo, int N, int ib,
                         PLASMA_Complex64_t *A, int LDA,
                         PLASMA_Complex64_t *WORK, int LDWORK,
                         int *INFO)
{
    int i, k, sb;
    PLASMA_Complex64_t alpha;
    static PLASMA_Complex64_t zone  = (PLASMA_Complex64_t) 1.0;

    /* Check input arguments */
    if (LDA < N) {
        coreblas_error(2, "Illegal value of N or LDA: LDA < N.");
        *INFO = -1;
        return;
    }

    *INFO = 0;

    /* Quick return */
    if (N == 1) {
        return;
    }

    if ( uplo == PlasmaLower ) {
        for(i = 0; i < N; i += ib) {
            sb = min(N-i, ib);

            /* Factorize the diagonal block */
            *INFO = CORE_zhetf3_nopiv(uplo, sb, &A[LDA*i+i], LDA);
            if (*INFO != 0) return;

            if ( i + sb < N ) {
                int height = N - i - sb;

                /* Solve the lower panel ( L21*D11 )*/
                cblas_ztrsm(
                    CblasColMajor, CblasRight, CblasLower,
                    CblasConjTrans, CblasUnit, height, sb, 
                    CBLAS_SADDR(zone), &A[LDA*i+i], LDA,
                    &A[LDA*i+i+sb], LDA);

                /* Scale the block to divide by D */
                for (k=0; k<sb; k++) {
                    alpha = zone / A[LDA*(i+k)+i+k];
                    cblas_zscal(height, CBLAS_SADDR(alpha), &A[LDA*(i+k)+i+sb], 1);
                }

                /* Update the trailing submatrix A22 = A22 - A21 * D11 * A21' */
                CORE_zhedrk(
                    PlasmaLower, PlasmaNoTrans, N-i-sb, sb, ib,
                    -1.0, &A[LDA* i+i+sb], LDA, 
                    1.0, &A[LDA*(i+sb)+i+sb], LDA,
                    &A[LDA*i+i], LDA+1, WORK, LDWORK);
            }
        }
    } else {
        for(i = ((N-1) / ib)*ib; i > -1; i -= ib) {
            sb = min(N-i, ib);

            /* Factorize the diagonal block */
            *INFO = CORE_zhetf3_nopiv(uplo, sb, &A[LDA*i+i], LDA);
            if (*INFO != 0) return;

            if ( N-i-sb > 0 ) {
                /* Solve the Upper panel ( U12*D22 ) */
                cblas_ztrsm(
                    CblasColMajor, CblasLeft, CblasUpper,
                    CblasConjTrans, CblasUnit, i, sb, 
                    CBLAS_SADDR(zone), &A[LDA*i+i], LDA, &A[LDA*i   ], LDA);

                /* Scale the block to divide by D */
                for (k=0; k<sb; k++) {
                    alpha = zone / A[LDA*(i+k)+i+k];
                    cblas_zscal(i, CBLAS_SADDR(alpha), &A[LDA*(i+k)], 1);
                }

                /* Update the trailing submatrix A22 = A22 - A21 * D11 * A21' */
                CORE_zhedrk(
                    PlasmaUpper, PlasmaNoTrans, i, sb, ib,
                    -1.0, &A[LDA*i], LDA, 1.0, A, LDA,
                    &A[LDA*i+i], LDA+1, WORK, LDWORK);
            }
        }
    }
}


/***************************************************************************//**
 *
 *  ZHETF1_NOPIV computes the factorization of a complex Hermitian matrix A
 *  without pivoting:
 *
 *                  A = L*D*L**H
 *
 *  where L is a product of permutation and unit lower triangular matrices,
 *  L**H is the conjugate transpose of L, and D is diagonal.
 *
 *  This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *  Golub & Van Loan, Alg 4.1.2 page 139
 **/
int CORE_zhetf1_nopiv(PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA)
{

    static PLASMA_Complex64_t zone = 1.0;
    static PLASMA_Complex64_t mzone=-1.0;

    int i, j, k, incx, incy, incx2, INFO=0;
    PLASMA_Complex64_t alpha;
    PLASMA_Complex64_t *v = (PLASMA_Complex64_t *)malloc(N*sizeof(PLASMA_Complex64_t));

    (void)uplo;
    incx2 = 1;

    j = 0;
    v[j] = A[j];
    A[j] = v[j];
    alpha = zone / v[j];
    cblas_zscal(N-(j+1), CBLAS_SADDR(alpha), &A[j+1], incx2);

    for (j = 1; j < N; j++) {

        /* Compute v(1:j) */
        for (i = 0; i < j; i++) {
            v[i] = A[LDA*i+j] * A[LDA*i+i];
        }
#ifdef COMPLEX
#else
        v[j] = A[LDA*j+j] - cblas_ddot(j, &A[j], N, v, 1);

        if (cabs(v[j]) < LAPACKE_dlamch_work('e')) {
            INFO = 1;
            fprintf(stderr,"CORE_zhetf1_nopiv: v[%i] < eps. Returning.\n",j);
            return INFO;
        }
#endif

        /* Store d(j) */
        A[LDA*j+j] = v[j];

        /* Compute L(j+1:n,j) */
        if ( (j+1) < N) {
            k = LDA*j+j+1;
            incx  = 1;
            incy  = 1;
            cblas_zgemv(CblasColMajor, CblasNoTrans, N-(j+1), j,
                        CBLAS_SADDR(mzone), &A[j+1], N, v, incx,
                        CBLAS_SADDR(zone), &A[k], incy);
            alpha = zone / v[j];
            cblas_zscal(N-(j+1), CBLAS_SADDR(alpha), &A[N*j+j+1], incx2);
        }

    }
    free(v);
    return INFO;
}

/***************************************************************************//**
 *
 *  ZHETF3_NOPIV computes a partial factorization of a complex Hermitian
 *  matrix A without pivoting. The partial factorization has the form:
 *
 *  A  =  ( I  U12 ) ( A11  0  ) (  I      0     )  if UPLO = 'U', or:
 *        ( 0  U22 ) (  0   D  ) ( U12**H U22**H )
 *
 *  A  =  ( L11  0 ) (  D   0  ) ( L11**H L21**H )  if UPLO = 'L'
 *        ( L21  I ) (  0  A22 ) (  0      I     )
 *
 *  ZHETF3_NOPIV is an auxiliary routine called by ZHETRF. It uses blocked code
 *  (calling Level 3 BLAS) to update the submatrix A11 (if UPLO = 'U') or
 *  A22 (if UPLO = 'L').
 *
 **/
int CORE_zhetf3_nopiv(PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA)
{
    /* Quick return */
    if (N==1)
        return 0;

    if (LDA < N) {
        coreblas_error(2, "Illegal value of N or LDA: LDA < N.");
        return -1;
    }

    /**/
    int k, info = 0;
    PLASMA_Complex64_t *Ak1k, *A1k;
    double Akk, alpha;
    
    if ( uplo == PlasmaLower )
    {
        /* Diagonal element */
        Akk  = creal(*A);
        /* Pointer on first extra diagonal element */
        Ak1k = A + 1;

        for( k=N-1; k>0; k--) {
            if ( fabs(Akk) < LAPACKE_dlamch_work('e') ) {
                info = k;
                fprintf(stderr, "CORE_zhetf3_nopiv: A(%d, %d) = %e < eps = %e. Returning.\n", N-k, N-k, Akk, LAPACKE_dlamch_work('e'));
                return info;
            }

            alpha = (double)1.0 / Akk;
            cblas_zdscal(k, alpha, Ak1k, 1);
            alpha = - Akk;
            cblas_zher(
                CblasColMajor, CblasLower, k, 
                alpha, Ak1k,       1, 
                       Ak1k + LDA, LDA);
            
            /* Move to next diagonal element */
            Ak1k += LDA; 
            Akk   = creal(*Ak1k); 
            Ak1k++;
        }
    }
    else {

        /* Pointer on column k */
        A1k = &A[LDA*(N-1)];

        for( k=N-1; k>0; k--) {

            /* Diagonal element */
            Akk = creal(A1k[k]); 
            
            if ( fabs(Akk) < LAPACKE_dlamch_work('e') ) {
                fprintf(stderr, "CORE_zhetf3_nopiv: A(%d, %d) = %e < eps. Returning.\n", k, k, Akk);
                return k;
            }

            alpha = (double)1.0 / Akk;
            cblas_zdscal(k, alpha, A1k, 1);
            alpha = - Akk;
            cblas_zher(
                CblasColMajor, CblasUpper, k, 
                alpha, A1k, 1, 
                       A,   LDA);
            
            /* Move to next diagonal element */
            A1k -= LDA; 
        }
    }
    return info;
}
