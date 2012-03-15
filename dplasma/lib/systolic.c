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


/*
 * Subtree for low-level
 */
static void dplasma_low_flat_init(     qr_subpiv_t *arg);

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
    int p = arg->p;
    int sq_p = sqrt(p);

    myassert( p = sq_p * sq_p);

    return min( p, gmt - k);
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
 *      2 - if m is reduced thanks to the 1st coordinate flat tree
 */
int dplasma_qr_gettype( const qr_piv_t *arg, const int k, const int m ) {
    int p = arg->p;
    int sq_p = sqrt(p);

    myassert(p = sq_p * sq_p);

    /* Element to be reduce with a single pivot */
    if (m < k + sq_p) 
        return 2;

    /* Element to be reduced with sq_p pivot */
    else if ( m < k + p )
        return 1;

    /* Local eliminations with a TS kernel */
    else return 0;
}

/****************************************************
 *                 DPLASMA_LOW_FLAT_TREE
Low Flat will reduce the tiles locally using a TS until there remains arg->p tiles.
The type of the tiles reduced by Low Flat is 0.
 ***************************************************/
/* Return the pivot to use for the row m at step k */
static int dplasma_low_flat_currpiv(const qr_subpiv_t *arg, const int m, const int k)
{
    myassert(m < k+ arg->p);
    return k + arg->p - 1 - m%(arg->p) ;
};

/* Return the next row which will use the row "pivot" as a pivot in step k after it has been used by row start ;  
  - initiated (have not killed anyone yet) with start==arg->ldd 
  - finishes (no more tiles to kill) with return arg->ldd */
static int dplasma_low_flat_nextpiv(const qr_subpiv_t *arg, const int pivot, const int k, const int start)
{
    if ( start == arg->ldd )
        return min( pivot + arg->p , arg->ldd );
    else
        return min( start + arg->p , arg->ldd );
}

/* Return the last row which has used the row "pivot" as a pivot in step k before the row start */
static int dplasma_low_flat_prevpiv(const qr_subpiv_t *arg, const int pivot, const int k, const int start)
{
    if ( start == arg->ldd )
        return m + arg->p * ((arg->ldd-1-k)/arg->p);
    else { 
        if ( start > pivot + arg->p )
            return start - arg->p;
        else
            return arg->ldd;
    }
};

static void dplasma_low_flat_init(qr_subpiv_t *arg){
    arg->currpiv = dplasma_low_flat_currpiv;
    arg->nextpiv = dplasma_low_flat_nextpiv;
    arg->prevpiv = dplasma_low_flat_prevpiv;
    arg->ipiv = NULL;
};


/****************************************************
 *                 DPLASMA_HIGH_FLAT_FLAT_TREE
Notation: 
In High_FlatFlat Tree we consider that there is only sqrt(p) processors (arg->p = sqrt(p), 
where p is the number of processors considered in Low_Flat Tree).
There are two possibilities for every function which will need to be considered, 
either we are in the first dimension elimination (p killers), 
either we are in the second dimension (a final killer).
 ***************************************************/

/* Return the pivot to use for the row m at step k */
static int dplasma_high_flat_flat_currpiv(const qr_subpiv_t *arg, const int m, const int k)
{
    if ( (m-k) % (arg->p) == 0 )
        return k;                                                                  //second dimension elimination
    else
        return k + (arg->p) * ( m - k )/(arg->p);                  //first dimension elimination
};

/* Return the next row which will use the row "pivot" as a pivot in step k after it has been used by row start ;  
  - initiated (have not killed anyone yet) with start==arg->ldd 
  - finishes (no more tiles to kill) with return arg->ldd */
static int dplasma_high_flat_flat_nextpiv(const qr_subpiv_t *arg, const int pivot, const int k, const int start)
{
    if ( (pivot-k) % (arg->p) == 0 && arg->ldd > 1 ) {
        if ( start == arg->ldd )
            return pivot+1;
        else 
            if ( start < min(k + (arg->p * arg-> p) , arg->ldd) && (start - pivot< arg->p - 1) ) //first dimension elimination
               return start+1;
            else 
                if (pivot == k && start == k + arg->p -1)  //when we step from the first dimension elimination to the second dimension elimination.
                    return k + arg->p;
                else
                    if (pivot == k && start == k + (arg->p) * (arg->p -1)) //the end of the second dimension elimination
                        return arg->ldd;
                    else
                        if (pivot == k && (start - k) % (arg->p) == 0 ) //second dimension elimination
                            return min( start + arg->p , arg->ldd );
    }
    return arg->ldd;
};

/* Return the last row which has used the row "pivot" as a pivot in step k before the row start */
static int dplasma_high_flat_flat_prevpiv(const qr_subpiv_t *arg, const int pivot, const int k, const int start)
{
    if ( pivot == k && arg->ldd > 1){
        if ( start == arg->ldd)
            return min( k + arg->p * (arg->p - 1) , k + arg->p * (arg->ldd - 1 - k)/arg->p );
        else
            if ( start == pivot + arg->p ) 
                return start - 1;
            else 
                if ( (start - pivot) % (arg->p) == 0)
                    return start - arg->p;
                else
                    if ( start - pivot > 1)
                        return start - 1;
    }
    else
        if ( start == arg->ldd )
            return pivot+ arg->p - 1;
        else 
            if ( start - pivot > 1 ) //first dimension elimination
               return start-1;
    return arg->ldd;
};

static void dplasma_high_flat_flat_init(qr_subpiv_t *arg){
    arg->currpiv = dplasma_high_flat_flat_currpiv;
    arg->nextpiv = dplasma_high_flat_flat_nextpiv;
    arg->prevpiv = dplasma_high_flat_flat_prevpiv;
    arg->ipiv = NULL;
};


/****************************************************
 *
 * Initialize/Finalize functions
 *
 ***************************************************/
qr_piv_t *dplasma_pivgen_init( tiled_matrix_desc_t *A,
                               int type_llvl, int type_hlvl,
                               int a, int p,
                               int domino, int tsrr )
{
    int low_mt, minMN;
    qr_piv_t *qrpiv = (qr_piv_t*) malloc( sizeof(qr_piv_t) );

/*    a = max( a, 1 );*/
    p = max( p, 1 );
    sq_p = sqrt(p);

    myassert( p == sq_p * qs_p);

    qrpiv->desc = A;
/*    qrpiv->a = a;*/
    qrpiv->p = p;
/*    qrpiv->domino = NULL;*/
/*    qrpiv->tsrr = NULL;*/
/*    qrpiv->perm = NULL;*/

    qrpiv->llvl = (qr_subpiv_t*) malloc( sizeof(qr_subpiv_t) );
    qrpiv->hlvl = NULL;

    minMN = min(A->mt, A->nt);
    low_mt = A->mt;

    qrpiv->llvl->minMN  = minMN;
    qrpiv->llvl->ldd    = low_mt;
/*    qrpiv->llvl->a      = a;*/
    qrpiv->llvl->p      = p;
/*    qrpiv->llvl->domino = domino;*/

    switch( type_llvl ) {
    case DPLASMA_FLAT_TREE :
    default:
        dplasma_low_flat_init(qrpiv->llvl);
    }

    if ( p > 1 ) {
        qrpiv->hlvl = (qr_subpiv_t*) malloc( sizeof(qr_subpiv_t) );

        qrpiv->hlvl->minMN  = minMN; //a priori on ne l'utilise jamais ?
        qrpiv->hlvl->ldd    = A->mt;
/*        qrpiv->hlvl->a      = a;*/
        qrpiv->hlvl->p      = sq_p;
/*        qrpiv->hlvl->domino = domino;*/

        switch( type_hlvl ) {
        case DPLASMA_FLAT_TREE :
        default:
            dplasma_high_flat_flat_init(qrpiv->hlvl);
        }
    }

    dplasma_qr_genperm( qrpiv );
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
