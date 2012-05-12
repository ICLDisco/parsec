/**
 *
 * @file core_zgetrf_rectil.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.4.6
 * @author Hatem Ltaief
 * @author Mathieu Faverge
 * @author Piotr Luszczek
 * @date 2009-11-15
 *
 * @precisions normal z -> c d s
 *
 **/

#include <math.h>
#include <cblas.h>
#include <lapacke.h>
#include <plasma.h>
#include "dplasma_cores.h"
#include "dplasma_zcores.h"

#ifndef min
#define min(__a, __b) ( ((__a) < (__b)) ? (__a) : (__b) )
#endif

static double sfmin;

static inline void
CORE_zgetrf_rectil_rec(const PLASMA_desc A, int *IPIV, int *info,
                       PLASMA_Complex64_t *pivot,
                       const int column, const int width,
                       const int ft,     const int lt);
static void
CORE_zgetrf_rectil_update(const PLASMA_desc A, int *IPIV,
                          const int column, const int n1,     const int n2,
                          const int ft,     const int lt);

/***************************************************************************//**
 *
 * @ingroup CORE_PLASMA_Complex64_t
 *
 *  CORE_zgetrf_rectil_1thrd computes a LU factorization of a general M-by-N
 *  matrix A stored in CCRB layout using partial pivoting with row
 *  interchanges.
 *
 *  The factorization has the form
 *
 *    A = P * L * U
 *
 *  where P is a permutation matrix, L is lower triangular with unit
 *  diagonal elements (lower trapezoidal if m > n), and U is upper
 *  triangular (upper trapezoidal if m < n).
 *
 *  This is the recursive version of the algorithm applied on tile layout.
 *
 *  WARNINGS:
 *     - The matrix A cannot be more than one tile wide.
 *
 *******************************************************************************
 *
 *  @param[in,out] A
 *          PLASMA descriptor of the matrix A to be factorized.
 *          On entry, the M-by-N matrix to be factorized.
 *          On exit, the factors L and U from the factorization
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 *  @param[out] IPIV
 *          The pivot indices; for 0 <= i < min(M,N) stored in Fortran
 *          mode (starting at 1), row i of the matrix was interchanged
 *          with row IPIV(i).
 *          On exit, each value IPIV[i] for 0 <= i < min(M,N) is
 *          increased by A.i, which means A.i < IPIV[i] <= A.i+M.
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval -k, the k-th argument had an illegal value
 *          \retval k if U(k,k) is exactly zero. The factorization
 *                  has been completed, but the factor U is exactly
 *                  singular, and division by zero will occur if it is used
 *                  to solve a system of equations.
 *
 */
int CORE_zgetrf_rectil_1thrd(const PLASMA_desc A, int *IPIV)
{
    int minMN = min( A.m, A.n );
    int info = 0;
    PLASMA_Complex64_t pivot;

    sfmin =  LAPACKE_dlamch_work('S');

    if ( A.nt > 1 ) {
        /*coreblas_error(1, "Illegal value of A.nt");*/
        return -1;
    }

    CORE_zgetrf_rectil_rec( A, IPIV, &info, &pivot,
                            0, minMN, 0, A.mt);

    if ( A.n > minMN ) {
        CORE_zgetrf_rectil_update( A, IPIV,
                                   0, minMN, A.n-minMN,
                                   0, A.mt);
    }

    return info;
}

#define A(m, n) BLKADDR(A, PLASMA_Complex64_t, m, n)

static inline void
CORE_zgetrf_rectil_update(const PLASMA_desc A, int *IPIV,
                          const int column, const int n1,     const int n2,
                          const int ft,     const int lt)
{
    int ld, lm, tmpM;
    int ip, j, it, i, ldft;
    PLASMA_Complex64_t zone  = 1.0;
    PLASMA_Complex64_t mzone = -1.0;
    PLASMA_Complex64_t *Atop, *Atop2, *U, *L;
    int offset = A.i;

    ldft = BLKLDD(A, 0);
    Atop  = A(0, 0) + column * ldft;
    Atop2 = Atop    + n1     * ldft;

    /* Swap to the right */
    int *lipiv = IPIV+column;
    int idxMax = column+n1;
    for (j = column; j < idxMax; ++j, ++lipiv) {
        ip = (*lipiv) - offset - 1;
        if ( ip != j )
        {
            it = ip / A.mb;
            i  = ip % A.mb;
            ld = BLKLDD(A, it);
            cblas_zswap(n2, Atop2                     + j, ldft,
                        A(it, 0) + (column+n1)*ld + i, ld   );
        }
    }

    /* Trsm on the upper part */
    U = Atop2 + column;
    cblas_ztrsm( CblasColMajor, CblasLeft, CblasLower,
                 CblasNoTrans, CblasUnit,
                 n1, n2, CBLAS_SADDR(zone),
                 Atop  + column, ldft,
                 U,              ldft );

    /* First tile */
    L = Atop + column + n1;
    tmpM = min(ldft, A.m) - column - n1;

    /* Apply the GEMM */
    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                 tmpM, n2, n1,
                 CBLAS_SADDR(mzone), L,      ldft,
                 U,      ldft,
                 CBLAS_SADDR(zone),  U + n1, ldft );


    /* Update the other blocks */
    for( it = ft+1; it < lt; it++)
    {
        ld = BLKLDD( A, it );
        L  = A( it, 0 ) + column * ld;
        lm = it == A.mt-1 ? A.m - it * A.mb : A.mb;

        /* Apply the GEMM */
        cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                     lm, n2, n1,
                     CBLAS_SADDR(mzone), L,          ld,
                                         U,          ldft,
                     CBLAS_SADDR(zone),  L + n1*ld,  ld );
    }
}

static void
CORE_zgetrf_rectil_rec(const PLASMA_desc A, int *IPIV, int *info,
                       PLASMA_Complex64_t *pivot,
                       const int column, const int width,
                       const int ft,     const int lt)
{
    int ld, jp, n1, n2, lm, tmpM, piv_sf;
    int ip, j, it, i, ldft;
    int max_i, max_it;
    PLASMA_Complex64_t zone  = 1.0;
    PLASMA_Complex64_t mzone = -1.0;
    PLASMA_Complex64_t tmp1;
    PLASMA_Complex64_t tmp2;
    PLASMA_Complex64_t pivval;
    PLASMA_Complex64_t *Atop, *Atop2, *U, *L;
    double             abstmp1;
    int offset = A.i;

    ldft = BLKLDD(A, 0);
    Atop = A(0, 0) + column * ldft;

    if ( width > 1 ) {
        /* Assumption: N = min( M, N ); */
        n1 = width / 2;
        n2 = width - n1;

        Atop2 = Atop + n1 * ldft;

        CORE_zgetrf_rectil_rec( A, IPIV, info, pivot,
                                column, n1, ft, lt );

        if ( *info != 0 )
            return;

        /* Swap to the right */
        {
            int *lipiv = IPIV+column;
            int idxMax = column+n1;
            for (j = column; j < idxMax; ++j, ++lipiv) {
                ip = (*lipiv) - offset - 1;
                if ( ip != j )
                {
                    it = ip / A.mb;
                    i  = ip % A.mb;
                    ld = BLKLDD(A, it);
                    cblas_zswap(n2, Atop2                     + j, ldft,
                                    A(it, 0) + (column+n1)*ld + i, ld   );
                }
            }
        }

        /* Trsm on the upper part */
        U = Atop2 + column;
        cblas_ztrsm( CblasColMajor, CblasLeft, CblasLower,
                     CblasNoTrans, CblasUnit,
                     n1, n2, CBLAS_SADDR(zone),
                     Atop  + column, ldft,
                     U,              ldft );

        /* SIgnal to other threads that they can start update */
        pivval = *pivot;
        if ( pivval == 0.0 ) {
            *info = column+n1;
            return;
        } else {
            if ( cabs(pivval) >= sfmin ) {
                piv_sf = 1;
                pivval = 1.0 / pivval;
            } else {
                piv_sf = 0;
            }
        }

        /* First tile */
        {
            L = Atop + column + n1;
            tmpM = min(ldft, A.m) - column - n1;

            /* Scale last column of L */
            if ( piv_sf ) {
                cblas_zscal( tmpM, CBLAS_SADDR(pivval), L+(n1-1)*ldft, 1 );
            } else {
                int i;
                Atop2 = L+(n1-1)*ldft;
                for( i=0; i < tmpM; i++, Atop2++)
                    *Atop2 = *Atop2 / pivval;
            }

            /* Apply the GEMM */
            cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                         tmpM, n2, n1,
                         CBLAS_SADDR(mzone), L,      ldft,
                                             U,      ldft,
                         CBLAS_SADDR(zone),  U + n1, ldft );

            /* Search Max in first column of U+n1 */
            tmp2    = U[n1];
            max_it  = ft;
            max_i   = cblas_izamax( tmpM, U+n1, 1 ) + n1;
            tmp1    = U[max_i];
            abstmp1 = cabs(tmp1);
            max_i  += column;
        }

        /* Update the other blocks */
        for( it = ft+1; it < lt; it++)
        {
            ld = BLKLDD( A, it );
            L  = A( it, 0 ) + column * ld;
            lm = it == A.mt-1 ? A.m - it * A.mb : A.mb;

            /* Scale last column of L */
            if ( piv_sf ) {
                cblas_zscal( lm, CBLAS_SADDR(pivval), L+(n1-1)*ld, 1 );
            } else {
                int i;
                Atop2 = L+(n1-1)*ld;
                for( i=0; i < lm; i++, Atop2++)
                    *Atop2 = *Atop2 / pivval;
            }

            /* Apply the GEMM */
            cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
                         lm, n2, n1,
                         CBLAS_SADDR(mzone), L,          ld,
                                             U,          ldft,
                         CBLAS_SADDR(zone),  L + n1*ld,  ld );

            /* Search the max on the first column of L+n1*ld */
            jp = cblas_izamax( lm, L+n1*ld, 1 );
            if ( cabs( L[n1*ld+jp] ) > abstmp1 ) {
                tmp1 = L[n1*ld+jp];
                abstmp1 = cabs(tmp1);
                max_i  = jp;
                max_it = it;
            }
        }

        jp = offset + max_it*A.mb + max_i;
        /* CORE_zamax1_thread( tmp1, thidx, thcnt, &thwin, */
        /*                     &tmp2, pivot, jp + 1, IPIV + column + n1 ); */
        IPIV[ column + n1 ] = jp + 1;
        *pivot = tmp1;

        U[n1] = *pivot; /* all threads have the pivot element: no need for synchronization */
        if ( jp-offset != column+n1 ) /* if there is a need to exchange the pivot */
        {
            ld = BLKLDD(A, max_it);
            Atop2 = A( max_it, 0 ) + (column + n1 )* ld + max_i;
            *Atop2 = tmp2;
        }

        CORE_zgetrf_rectil_rec( A, IPIV, info, pivot,
                                column+n1, n2, ft, lt );
        if ( *info != 0 )
            return;

        /* Swap to the left */
        {
            int *lipiv = IPIV+column+n1;
            int idxMax = column+width;
            for (j = column+n1; j < idxMax; ++j, ++lipiv) {
                ip = (*lipiv) - offset - 1;
                if ( ip != j )
                {
                    it = ip / A.mb;
                    i  = ip % A.mb;
                    ld = BLKLDD(A, it);
                    cblas_zswap(n1, Atop + j,                 ldft,
                                    A(it, 0) + column*ld + i, ld  );
                }
            }
        }
    }
    else if ( width == 1 ) {

        /* Search maximum for column 0 */
        if ( column == 0 )
        {
            tmp2 = Atop[column];

            /* First tmp1 */
            ld = BLKLDD(A, ft);
            Atop2   = A( ft, 0 );
            lm      = ft == A.mt-1 ? A.m - ft * A.mb : A.mb;
            max_it  = ft;
            max_i   = cblas_izamax( lm, Atop2, 1 );
            tmp1    = Atop2[max_i];
            abstmp1 = cabs(tmp1);

            /* Update */
            for( it = ft+1; it < lt; it++)
            {
                Atop2= A( it, 0 );
                lm   = it == A.mt-1 ? A.m - it * A.mb : A.mb;
                jp   = cblas_izamax( lm, Atop2, 1 );
                if (  cabs(Atop2[jp]) > abstmp1 ) {
                    tmp1 = Atop2[jp];
                    abstmp1 = cabs(tmp1);
                    max_i  = jp;
                    max_it = it;
                }
            }

            jp = offset + max_it*A.mb + max_i;
            /* CORE_zamax1_thread( tmp1, thidx, thcnt, &thwin, */
            /*                     &tmp2, pivot, jp + 1, IPIV + column ); */
            IPIV[column] = jp +1;
            *pivot = tmp1;
            Atop[0] = *pivot; /* all threads have the pivot element: no need for synchronization */

            if ( jp-offset != 0 ) /* if there is a need to exchange the pivot */
            {
                Atop2 = A( max_it, 0 ) + max_i;
                *Atop2 = tmp2;
            }
        }

        /* If it is the last column, we just scale */
        if ( column == (min(A.m, A.n))-1 ) {

            pivval = *pivot;
            if ( pivval != 0.0 ) {
                if ( cabs(pivval) >= sfmin ) {
                    pivval = 1.0 / pivval;

                    /*
                     * We guess than we never enter the function with m == A.mt-1
                     * because it means that there is only one thread
                     */
                    lm = ft == A.mt-1 ? A.m - ft * A.mb : A.mb;
                    cblas_zscal( lm - column - 1, CBLAS_SADDR(pivval), Atop+column+1, 1 );

                    for( it = ft+1; it < lt; it++)
                    {
                        ld = BLKLDD(A, it);
                        Atop2 = A( it, 0 ) + column * ld;
                        lm = it == A.mt-1 ? A.m - it * A.mb : A.mb;
                        cblas_zscal( lm, CBLAS_SADDR(pivval), Atop2, 1 );
                    }
                } else {
                    /*
                     * We guess than we never enter the function with m == A.mt-1
                     * because it means that there is only one thread
                     */
                    int i;
                    Atop2 = Atop + column + 1;
                    lm = ft == A.mt-1 ? A.m - ft * A.mb : A.mb;

                    for( i=0; i < lm-column-1; i++, Atop2++)
                        *Atop2 = *Atop2 / pivval;

                    for( it = ft+1; it < lt; it++)
                        {
                            ld = BLKLDD(A, it);
                            Atop2 = A( it, 0 ) + column * ld;
                            lm = it == A.mt-1 ? A.m - it * A.mb : A.mb;

                            for( i=0; i < lm; i++, Atop2++)
                                *Atop2 = *Atop2 / pivval;
                        }
                }
            }
            else {
                *info = column + 1;
                return;
            }
        }
    }
}

