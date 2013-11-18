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
 *
 */
#include <dague.h>
#include <core_blas.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"
#include "dplasma_qr_pivgen.h"

#include <math.h>
#if defined(HAVE_STRING_H)
#include <string.h>
#endif  /* defined(HAVE_STRING_H) */

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };
static inline int dague_imax(int a, int b) { return (a >= b) ? a : b; };

#define PRINT_PIVGEN 0
#ifdef PRINT_PIVGEN
#define myassert( test ) {if ( ! (test) ) return -1;}
#else
#define myassert(test) {assert((test)); return -1;}
#endif

#define nbextra1_formula ( (k % pa) > (pa - p) ) ? (-k)%pa + pa : 0

static int systolic_getnbgeqrf( const dplasma_qrtree_t *qrtree, int k )
{
    int pq = qrtree->p * qrtree->a;
    return dague_imin( pq, qrtree->desc->mt - k);
}

static int systolic_getm( const dplasma_qrtree_t *qrtree, int k, int i )
{
    (void)qrtree;
    return k+i;
}

static int systolic_geti( const dplasma_qrtree_t *qrtree, int k, int m )
{
    (void)qrtree;
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
static int systolic_gettype( const dplasma_qrtree_t *qrtree, int k, int m ) {
    int p = qrtree->p;
    int q = qrtree->a;
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
static int systolic_currpiv(const dplasma_qrtree_t *qrtree, int k, int m)
{
    int p = qrtree->p;
    int q = qrtree->a;
    int pq = p * q;

    switch( systolic_gettype( qrtree, k, m ) )
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
        return qrtree->desc->mt;
    }
};

/**
 *  systolic_nextpiv - Computes the next row killed by the row p, after
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
static int systolic_nextpiv(const dplasma_qrtree_t *qrtree, int k, int pivot, int start)
{
    int ls, lp, nextp;
    int q = qrtree->a;
    int p = qrtree->p;
    int pq = p * q;
    int mt = qrtree->desc->mt;

    myassert( start > pivot && pivot >= k );
    myassert( start == mt || pivot == systolic_currpiv( qrtree, k, start ) );

    /* TS level common to every case */
    ls = (start < mt) ? systolic_gettype( qrtree, k, start ) : -1;
    lp = systolic_gettype( qrtree, k, pivot );

    switch( ls )
        {
        case -1:

            if ( lp == DPLASMA_QR_KILLED_BY_TS ) {
                myassert( start == mt );
                return mt;
            }

        case DPLASMA_QR_KILLED_BY_TS:

            if ( start == mt )
                nextp = pivot + pq;
            else
                nextp = start + pq;

            if ( nextp < mt )
                return nextp;

            start = mt;

        case DPLASMA_QR_KILLED_BY_LOCALTREE:

            if (lp < DPLASMA_QR_KILLED_BY_DISTTREE)
                return mt;

            if ( start == mt )
                nextp = pivot + p;
            else
                nextp = start + p;

            if ( (nextp >= k + p) &&
                 (nextp < k + pq) &&
                 (nextp < mt) )
                return nextp;

            start = mt;

        case DPLASMA_QR_KILLED_BY_DISTTREE:

            if (pivot > k)
                return mt;

            if ( start == mt )
                nextp = pivot + 1;
            else
                nextp = start + 1;

            if ( nextp < k + p )
                return nextp;

        default:
            return mt;
        }
}

/**
 *  systolic_prevpiv - Computes the previous row killed by the row p, before
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
static int systolic_prevpiv(const dplasma_qrtree_t *qrtree, int k, int pivot, int start)
{
    int ls, lp, nextp;
    int rpivot;
    int q = qrtree->a;
    int p = qrtree->p;
    int pq = p * q;
    int mt = qrtree->desc->mt;

    rpivot = pivot % pq; /* Staring index in this distribution               */

    myassert( start >= pivot && pivot >= k && start < mt );
    myassert( start == pivot || pivot == systolic_currpiv( qrtree, k, start ) );

    /* TS level common to every case */
    ls = systolic_gettype( qrtree, k, start );
    lp = systolic_gettype( qrtree, k, pivot );

    if ( lp == DPLASMA_QR_KILLED_BY_TS )
      return mt;

    myassert( lp >= ls );
    switch( ls )
        {
        case DPLASMA_QR_KILLED_BY_DISTTREE:

            if ( pivot == k ) {
                if ( start == pivot ) {
                    nextp = start + p-1;

                    while( pivot < nextp && nextp >= mt )
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
                           nextp >= mt )
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
                    nextp = mt - (mt - rpivot - 1)%pq - 1;

                    while( pivot < nextp && nextp >= mt )
                        nextp -= pq;
                } else {
                    nextp = start - pq;
                }
                assert(nextp < mt);
                if ( pivot < nextp )
                    return nextp;
            }

        default:
            return mt;
        }
};

/****************************************************
 *
 * Initialize/Finalize functions
 *
 ***************************************************/
void dplasma_systolic_init( dplasma_qrtree_t *qrtree,
                            tiled_matrix_desc_t *A,
                            int p, int q )
{
    qrtree->getnbgeqrf = systolic_getnbgeqrf;
    qrtree->getm       = systolic_getm;
    qrtree->geti       = systolic_geti;
    qrtree->gettype    = systolic_gettype;
    qrtree->currpiv    = systolic_currpiv;
    qrtree->nextpiv    = systolic_nextpiv;
    qrtree->prevpiv    = systolic_prevpiv;

    qrtree->desc = A;
    qrtree->a    = dague_imax( q, 1 );
    qrtree->p    = dague_imax( p, 1 );
    qrtree->args = NULL;

    return;
}

void dplasma_systolic_finalize( dplasma_qrtree_t *qrtree )
{
    if ( qrtree->args != NULL) {
        free( qrtree->args );
    }
}
