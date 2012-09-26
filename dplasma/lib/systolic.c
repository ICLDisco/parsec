/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 *
 * This file contains all the function to describe the dependencies
 * used in the Xgeqrf_param.jdf file.
 * The QR factorization done with this file relies on three levels:
 *     - the first one is using a flat tree with TS kernels. The
 *       height of this tree is defined by the parameter 'a'. If 'a'
 *       is set to A->mt, the factorization is identical to the one
 *       perform by PLASMA_zgeqrf.
 *       For all subdiagonal "macro-tiles", the line reduced is always
 *       the first.  For all diagonal "macro-tiles", the factorization
 *       performed is identical to the one performed by PLASMA_zgeqrf.
 *
 *     - the third level is using a reduction tree of size 'p'. By
 *       default, the parameter 'p' should be equal to the number of
 *       processors used for the computation, but can be set
 *       differently. (see further example). The type of tree used at
 *       this level is defined by the hlvl parameter. It can be flat
 *       or greedy.
 *       CODE DETAILS: This tree and all the function related to it
 *       are performing a QR factorization on a band matrix with 'p'
 *       the size of the band. All the functions take global indices
 *       as input and return global indices as output.
 *
 *     - Finally, a second 'low' level of reduction tree is applied.
 *       The size of this tree is induced by the parameters 'a' and 'p'
 *       from the first and third levels and is A->mt / ( p * a ). This
 *       tree is reproduced p times for each subset of tiles
 *       S_k = {i in [0, A->mt-1] \ i%p*a = k } with k in [0, p-1].
 *       The tree used for the reduction is defined by the llvl
 *       parameter and can be: flat, greedy, fibonacci or binary.
 *       CODE DETAILS: For commodity, the size of this tree is always
 *       ceil(A->mt / (p * a) ) inducing some extra tests in the code.
 *       All the functions related to this level of tree take as input
 *       the local indices in the A->mt / (p*a) matrix and the global
 *       k. They return the local index. The reductions are so
 *       performed on a trapezoidal matrices where the step is defined
 *       by a:
 *                                    <- min( lhlvl_mt, min( mt, nt ) ) ->
 *                                     __a__   a     a
 *                                    |     |_____
 *                                    |           |_____
 *                                    |                 |_____
 *        llvl_mt = ceil(MT/ (a*p))   |                       |_____
 *                                    |                             |_____
 *                                    |___________________________________|
 *
 *
 *
 *   At each step of the factorization, the lines of tiles are divided
 *   in 4 types:
 *     - QRPARAM_TILE_TS: They are the lines annihilated by a TS
 *     kernel, these lines are never used as an annihilator.  They are
 *     the lines i, with 1 < (i/p)%a < a and i > (k+1)*p
 *     - QRPARAM_TILE_LOCALTT: They are the lines used as annhilitor
 *     in the TS kernels annihiling the QRPARAM_TILE_TS lines.  They
 *     are themselves annihilated by the TT kernel of the low level
 *     reduction tree.  The roots of the local trees are the lines i,
 *     with i/p = k.
 *     - QRPARAM_TILE_DOMINO: These are the lines that are
 *     annhilihated with a domino effect in the band defined by (i/p)
 *     <= k and i >= k
 *     - QRPARAM_TILE_DISTTT: These are the lines annihilated by the
 *     high level tree to reduce communications.
 *     These lines are defined by (i-k)/p = 0.
 */
#include <dague.h>
#include <plasma.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"
#include "dplasma_qr_pivgen.h"

#include <math.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */

#ifndef min
#define min(__a, __b) ( ( (__a) < (__b) ) ? (__a) : (__b) )
#endif

#ifndef max
#define max(__a, __b) ( ( (__a) > (__b) ) ? (__a) : (__b) )
#endif

#define PRINT_PIVGEN 0
#ifdef PRINT_PIVGEN
#define myassert( test ) {if ( ! (test) ) return -1;}
#else
#define myassert(test) {assert((test)); return -1;}
#endif
#define VERSION 2

/*
 * Common functions
 */
int dplasma_qr_getnbgeqrf( const qr_piv_t *arg, const int k, const int gmt );
int dplasma_qr_getm(       const qr_piv_t *arg, const int k, const int i   );
int dplasma_qr_geti(       const qr_piv_t *arg, const int k, const int m   );
int dplasma_qr_gettype(    const qr_piv_t *arg, const int k, const int m   );

/* static void dplasma_qr_genperm   (       qr_piv_t *qrpiv ); */
/* static int  dplasma_qr_getinvperm( const qr_piv_t *qrpiv, const int k, int m ); */

/****************************************************
 *             Common ipiv
 ***************************************************
 *
 * Common prameters to the 4 following functions:
 *    a - Parameter a for the tunable QR
 *    p - Parameter p for the tunable QR
 *    k - Step k of the QR factorization
 *
 */

#define nbextra1_formula ( (k % pa) > (pa - p) ) ? (-k)%pa + pa : 0

/*
 * Extra parameter:
 *    gmt - Global number of tiles in a column of the complete distributed matrix
 * Return:
 *    The number of geqrt to execute in the panel k
 */
int dplasma_qr_getnbgeqrf( const qr_piv_t *arg, const int k, const int gmt ) {
    int pq = arg->p * arg->a;

    return min( pq, gmt - k);
}

/*
 * Extra parameter:
 *    i - indice of the geqrt in the continuous space
 * Return:
 *    The global indice m of the i th geqrt in the panel k
 */
int dplasma_qr_getm( const qr_piv_t *arg, const int k, const int i )
{
    (void) arg;
    return k+i;
}

/*
 * Extra parameter:
 *    m - The global indice m of a geqrt in the panel k
 * Return:
 *    The index i of the geqrt in the panel k
 */
int dplasma_qr_geti( const qr_piv_t *arg, const int k, int m )
{
    (void) arg;
    return m-k;
}

/*
 * Extra parameter:
 *      m - The global indice m of the row in the panel k
 * Return
 *     -1 - Error
 *      0 - if m is reduced thanks to a TS kernel
 *      1 - if m is reduced thanks to the 2nd coordinate flat tree
 *      3 - if m is reduced thanks to the 1st coordinate flat tree
 */
int dplasma_qr_gettype( const qr_piv_t *arg, const int k, const int m ) {
    int p = arg->p;
    int q = arg->a;
    int pq = p * q;

    /* Local eliminations with a TS kernel */
    if ( m >= k + pq )
        return 0;

    /* Element to be reduce with a single pivot */
    else if ( m >= k+p )
        return 1;

    /* Element to be reduced with sq_p pivot */
    else return 3;
}


/****************************************************
 *
 *   Generic functions currpiv,prevpiv,nextpiv
 *
 ***************************************************/
int dplasma_qr_currpiv(const qr_piv_t *arg, const int m, const int k)
{
    int p = arg->p;
    int q = arg->a;
    int pq = p * q;

    switch( dplasma_qr_gettype( arg, k, m ) )
    {
    case 0:
        return (m - k) % pq + k;
        break;
    case 1:
        return (m - k) % p + k;
        break;
    case 3:
        return k;
        break;
    default:
        return arg->desc->mt;
    }
};

/**
 *  dplasma_qr_nextpiv - Computes the next row killed by the row p, after
 *  it has kill the row start.
 *
 * @param[in] p
 *         Line used as killer
 *
 * @param[in] k
 *         Factorization step
 *
 * @param[in] start
 *         Starting point to search the next line killed by p after start
 *         start must be equal to A.mt to find the first row killed by p.
 *         if start != A.mt, start must be killed by p
 *
 * @return:
 *   - -1 if start doesn't respect the previous conditions
 *   -  m, the following row killed by p if it exists, A->mt otherwise
 */
int dplasma_qr_nextpiv(const qr_piv_t *arg, int pivot, const int k, int start)
{
    int ls, lp, nextp;
    int q = arg->a;
    int p = arg->p;
    int pq = p * q;

    myassert( start > pivot && pivot >= k );
    myassert( start == arg->desc->mt || pivot == dplasma_qr_currpiv( arg, start, k ) );

    /* TS level common to every case */
    ls = (start < arg->desc->mt) ? dplasma_qr_gettype( arg, k, start ) : -1;
    lp = dplasma_qr_gettype( arg, k, pivot );

    switch( ls )
        {
        case -1:

            if ( lp == DPLASMA_QR_KILLED_BY_TS ) {
                myassert( start == arg->desc->mt );
                return arg->desc->mt;
            }

        case DPLASMA_QR_KILLED_BY_TS:

            if ( start == arg->desc->mt )
                nextp = pivot + pq;
            else
                nextp = start + pq;

            if ( nextp < arg->desc->mt )
                return nextp;

            start = arg->desc->mt;

        case DPLASMA_QR_KILLED_BY_LOCALTREE:

            if (lp < DPLASMA_QR_KILLED_BY_DISTTREE)
                return arg->desc->mt;

            if ( start == arg->desc->mt )
                nextp = pivot + p;
            else
                nextp = start + p;

            if ( (nextp >= k + p) &&
                 (nextp < k + pq) &&
                 (nextp < arg->desc->mt) )
                return nextp;

            start = arg->desc->mt;

        case DPLASMA_QR_KILLED_BY_DISTTREE:

            if (pivot > k)
                return arg->desc->mt;

            if ( start == arg->desc->mt )
                nextp = pivot + 1;
            else
                nextp = start + 1;

            if ( nextp < k + p )
                return nextp;

        default:
            return arg->desc->mt;
        }
}

/**
 *  dplasma_qr_prevpiv - Computes the previous row killed by the row p, before
 *  to kill the row start.
 *
 * @param[in] p
 *         Line used as killer
 *
 * @param[in] k
 *         Factorization step
 *
 * @param[in] start
 *         Starting point to search the previous line killed by p before start
 *         start must be killed by p, and start must be greater or equal to p
 *
 * @return:
 *   - -1 if start doesn't respect the previous conditions
 *   -  m, the previous row killed by p if it exists, A->mt otherwise
 */
int dplasma_qr_prevpiv(const qr_piv_t *arg, int pivot, const int k, int start)
{
    int ls, lp, nextp;
    int rpivot;
    int q = arg->a;
    int p = arg->p;
    int pq = p * q;

    rpivot = pivot % pq; /* Staring index in this distribution               */

    myassert( start >= pivot && pivot >= k && start < arg->desc->mt );
    myassert( start == pivot || pivot == dplasma_qr_currpiv( arg, start, k ) );

    /* TS level common to every case */
    ls = dplasma_qr_gettype( arg, k, start );
    lp = dplasma_qr_gettype( arg, k, pivot );

    if ( lp == DPLASMA_QR_KILLED_BY_TS )
      return arg->desc->mt;

    myassert( lp >= ls );
    switch( ls )
        {
        case DPLASMA_QR_KILLED_BY_DISTTREE:

            if ( pivot == k ) {
                if ( start == pivot ) {
                    nextp = start + p-1;

                    while( pivot < nextp && nextp >= arg->desc->mt )
                        nextp--;
                } else {
                    nextp = start - 1;
                }

                if ( pivot < nextp &&
                     nextp < k + p )
                    return nextp;
            }
            start = pivot;

        case DPLASMA_QR_KILLED_BY_LOCALTREE:

            if ( lp > DPLASMA_QR_KILLED_BY_LOCALTREE ) {
                if ( start == pivot ) {
                    nextp = start + (q-1) * p;

                    while( pivot < nextp &&
                           nextp >= arg->desc->mt )
                        nextp -= p;
                } else {
                    nextp = start - p;
                }

                if ( pivot < nextp &&
                     nextp < k + pq )
                    return nextp;
            }
            start = pivot;

        case DPLASMA_QR_KILLED_BY_TS:
            /* Search for predecessor in TS tree */
            if ( lp > DPLASMA_QR_KILLED_BY_TS ) {
                if ( start == pivot ) {
                    nextp = arg->desc->mt - (arg->desc->mt - rpivot - 1)%pq - 1;

                    while( pivot < nextp && nextp >= arg->desc->mt )
                        nextp -= pq;
                } else {
                    nextp = start - pq;
                }
                assert(nextp < arg->desc->mt);
                if ( pivot < nextp )
                    return nextp;
            }

        default:
            return arg->desc->mt;
        }
};

/****************************************************
 *
 * Initialize/Finalize functions
 *
 ***************************************************/
qr_piv_t *dplasma_systolic_init( tiled_matrix_desc_t *A,
                                 int p, int q )
{
    qr_piv_t *qrpiv = (qr_piv_t*) malloc( sizeof(qr_piv_t) );

    qrpiv->desc = A;
    qrpiv->a = max( q, 1 );
    qrpiv->p = max( p, 1 );
    qrpiv->domino = 0;
    qrpiv->tsrr = 0;
    qrpiv->perm = NULL;
    qrpiv->llvl = NULL;
    qrpiv->hlvl = NULL;

    return qrpiv;
}

void dplasma_pivgen_finalize( qr_piv_t *qrpiv )
{
    if ( qrpiv->llvl != NULL) {
        if ( qrpiv->llvl->ipiv != NULL )
            free( qrpiv->llvl->ipiv );
        free( qrpiv->llvl );
    }

    if ( qrpiv->hlvl != NULL) {
        if ( qrpiv->hlvl->ipiv != NULL )
            free( qrpiv->hlvl->ipiv );
        free( qrpiv->hlvl );
    }

    if ( qrpiv->perm != NULL )
        free(qrpiv->perm);

    free(qrpiv);
}
