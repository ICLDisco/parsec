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
    int p    = arg->p;
    int sq_p = (int)sqrt((double)p);

    myassert( p == (sq_p * sq_p) );

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
 *      3 - if m is reduced thanks to the 1st coordinate flat tree
 */
int dplasma_qr_gettype( const qr_piv_t *arg, const int k, const int m ) {
    int p = arg->p;
    int sq_p = (int)sqrt((double)p);

    assert(p == sq_p * sq_p);

    /* Local eliminations with a TS kernel */
    if ( m > k + p -1 )
        return 0;

    /* Element to be reduce with a single pivot */
    else if ( (m - k) % sq_p == 0 )
        return 3;

    /* Element to be reduced with sq_p pivot */
    else return 1;
}

/****************************************************
 *                 DPLASMA_LOW_FLAT_TREE
Low Flat will reduce the tiles locally using a TS until there remains arg->p tiles.
The type of the tiles reduced by Low Flat is 0.
 ***************************************************/
/* Return the pivot to use for the row m at step k */
static int dplasma_low_flat_currpiv(const qr_subpiv_t *arg, const int m, const int k)
{
    int ord = (m - k)%(arg->p);
    if ( ord >= 0 )
        return k + ord ;
    else 
        return k + arg->p + ord;
};

/* Return the next row which will use the row "pivot" as a pivot in step k after it has been used by row start ;
  - initiated (have not killed anyone yet) with start==arg->ldd
  - finishes (no more tiles to kill) with return arg->ldd */
static int dplasma_low_flat_nextpiv(const qr_subpiv_t *arg, const int pivot, const int k, const int start)
{
    int sq_p = (int)sqrt((double)(arg->p));
    assert(arg->p == sq_p * sq_p);
    assert(pivot < k + arg->p);

    if ( start == arg->ldd ){
        if ( pivot + arg->p < arg->ldd )
            return pivot + arg->p;
        else
            if ( (pivot- k)%sq_p == 0 ) //type pivot = 3
                return pivot +1;
            else //type pivot = 1
                return arg->ldd;
        }
    else {
        if ( start + arg->p < arg->ldd )
            return start + arg->p;
        else
            if ( (k- pivot)%sq_p == 0 )  //type pivot = 3
                return pivot +1;
            else //type pivot = 1
                return arg->ldd;
        }
}

/* Return the last row which has used the row "pivot" as a pivot in step k before the row start */
static int dplasma_low_flat_prevpiv(const qr_subpiv_t *arg, const int pivot, const int k, const int start)
{
    int sq_p = (int)sqrt((double)(arg->p));
    myassert(arg->p == sq_p * sq_p);

    if ( start == arg->ldd ) { //n'arrive que lorsque pivot est une tile de type 1
        if ( (arg->ldd -1 - pivot)/arg->p == 0 ) //dans ce cas c'est inutile, il n'y a rien à faire.
            return arg->ldd ;
        else
            return pivot + ((arg->ldd -1 - pivot)/arg->p) * arg->p;
    }
    else {
        if ( start > pivot + arg->p ) //on passe de l'extermination d'une tile de type 0 à 0, ok.
            return start - arg->p;
        else {
            if ( start == pivot + arg->p )
                return arg->ldd;
            else {//on passe de l'extermination d'une tile de type 1 à 0
                if ( (pivot - k)%sq_p == 0 ) {
                    if ( (arg->ldd -1 - pivot)/arg->p == 0 ) //dans ce cas c'est inutile, il n'y a rien à faire.
                        return arg->ldd ;
                    else
                        return pivot + ((arg->ldd -1 - pivot)/arg->p) * arg->p;
                    }
                else
                    return arg->ldd;
            }
        }
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
    assert( m - k < arg->p * arg->p);
    if ( (m-k) % (arg->p) == 0 )
        return k;                                                                  //second dimension elimination
    else
        return k + ( ( m - k )/(arg->p) ) * (arg->p);                  //first dimension elimination
};

/* Return the next row which will use the row "pivot" as a pivot in step k after it has been used by row start ;
  - initiated (have not killed anyone yet) with start==arg->ldd
  - finishes (no more tiles to kill) with return arg->ldd */
static int dplasma_high_flat_flat_nextpiv(const qr_subpiv_t *arg, const int pivot, const int k, const int start)
{
    assert ( (pivot-k) % (arg->p) == 0 && pivot-k < arg->p * arg->p && arg->ldd > 1 );
    if ( start == arg->ldd )  //base case.
        return pivot+1;
   else
        if ( (start - pivot< arg->p - 1) ) //first dimension elimination before: "&& start < min(k + (arg->p * arg-> p) , arg->ldd)"
           return start+1;
        else
           if (pivot == k) {
                if (start == k + arg->p -1)  //when we step from the first dimension elimination to the second dimension elimination.
                   return k + arg->p;
                else
                    if ( start == k + (arg->p) * (arg->p -1) ) //the end of the second dimension elimination
                        return arg->ldd;
                    else
                        if ( (start - k) % (arg->p) == 0 ) //second dimension elimination
                            return min( start + arg->p , arg->ldd );
            }
            else return arg->ldd;
    return arg->ldd;
};

/* Return the last row which has used the row "pivot" as a pivot in step k before the row start */
static int dplasma_high_flat_flat_prevpiv(const qr_subpiv_t *arg, const int pivot, const int k, const int start)
{
    assert(arg->ldd >1);
    if ( (pivot-k)%(arg->p) != 0)
        return -5;


    if ( pivot == k ){
        if ( start == arg->ldd){ 
            if ( k + arg->p >= arg-> ldd ) //if true: no second dimension elimination.
                return min( k + arg->p -1, max( k+1, arg->ldd-1));
            else
                return min( k + arg->p * (arg->p - 1) , k + arg->p * ((arg->ldd - 1 - k)/arg->p) ); //else last element used in the second dimension elimination
            }
        else {
            if ( start == pivot + arg->p )
                return start - 1;
            else {
                if ( (start - pivot) % (arg->p) == 0)
                    return start - arg->p;
                else
                    if ( start - pivot > 1)
                        return start - 1;
                    else
                        return min( pivot + ((arg->ldd  - 1 - pivot)/(arg->p * arg->p)) * arg->p * arg->p, arg->ldd);
                }
            }
        }
    else //(pivot-k)%arg->p == 0, but not k.
        if ( start == arg->ldd )
            return min( pivot + arg->p - 1, max( arg->ldd -1, pivot +1) );
        else
            if ( start - pivot > 1 ) //first dimension elimination
               return start-1;
            else
                return min( pivot + ((arg->ldd -1- pivot)/(arg->p * arg->p)) * arg->p * arg->p, arg->ldd); //todo here

    if ( (pivot + (arg->p - 1) * arg->p) >= arg->ldd ) 
        return arg->ldd;
    else
        return pivot + (arg->p - 1) * arg->p;
};

static void dplasma_high_flat_flat_init(qr_subpiv_t *arg){
    arg->currpiv = dplasma_high_flat_flat_currpiv;
    arg->nextpiv = dplasma_high_flat_flat_nextpiv;
    arg->prevpiv = dplasma_high_flat_flat_prevpiv;
    arg->ipiv = NULL;
};

/****************************************************
 *
 *   Generic functions currpiv,prevpiv,nextpiv
 *
 ***************************************************/
int dplasma_qr_currpiv(const qr_piv_t *arg, const int m, const int k)
{
    /* TS level common to every case */

    switch( dplasma_qr_gettype( arg, k, m ) )
    {
    case 0:
        return arg->llvl->currpiv(arg->llvl, m, k) ;
        break;
    case 1:
        return arg->hlvl->currpiv(arg->hlvl, m, k) ;
        break;
    case 3:
        return arg->hlvl->currpiv(arg->hlvl, m, k);
        break;
    default:
        return arg->desc->mt;
    }
};

int dplasma_qr_nextpiv(const qr_piv_t *arg, int pivot, const int k, int start)
{
    int ls, lp;

    myassert( start > pivot && pivot >= k );
    myassert( start == arg->desc->mt || pivot == dplasma_qr_currpiv( arg, start, k ) ); // Returns -1 if pivot is not the pivot of start at step k.

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
            return arg->llvl->nextpiv(arg->llvl, pivot, k, start );
            break;

        case DPLASMA_QR_KILLED_BY_TS:
            return arg->llvl->nextpiv(arg->llvl, pivot, k, start );
            break;

        case DPLASMA_QR_KILLED_BY_LOCALTREE:
            myassert( arg->hlvl != NULL );
            return arg->hlvl->nextpiv(arg->hlvl, pivot, k, start );
            break;

        case DPLASMA_QR_KILLED_BY_DISTTREE:
            myassert( arg->hlvl != NULL );
            return arg->hlvl->nextpiv(arg->hlvl, pivot, k, start );
            break;

        default:
            return arg->desc->mt;
        }
}

int dplasma_qr_prevpiv(const qr_piv_t *arg, int pivot, const int k, int start)
{
    int ls, lp;
    int temp;

    myassert( start >= pivot && pivot >= k && start <= arg->desc->mt );
    myassert( start == arg->desc->mt || start == pivot || pivot == dplasma_qr_currpiv( arg, start, k ) );

    /* TS level common to every case */
    ls = (start < arg->desc->mt) ? dplasma_qr_gettype( arg, k, start ) : -1;
    lp = dplasma_qr_gettype( arg, k, pivot );

    if ( lp == DPLASMA_QR_KILLED_BY_TS ) // This is because Mathieu est un gros sale =)
      return arg->desc->mt;

    myassert( lp >= ls );
    switch( ls )
        {
        case -1:
            if ( lp == DPLASMA_QR_KILLED_BY_TS ) {
                myassert( start == arg->desc->mt );
                return -1;
            }
            temp = arg->hlvl->prevpiv(arg->hlvl, pivot, k, start );
            if (temp == -5)
                return arg->llvl->prevpiv(arg->llvl, pivot, k, start );
            else 
                return temp;
            break;

        case DPLASMA_QR_KILLED_BY_TS:
            return arg->llvl->prevpiv(arg->llvl, pivot, k, start );
            break;

        case DPLASMA_QR_KILLED_BY_LOCALTREE:
            if ( start == pivot && (pivot + arg->llvl->p >= arg->desc->mt) )
                return arg->desc->mt;
            else
                myassert( start != pivot );
            temp = arg->hlvl->prevpiv(arg->hlvl, pivot, k, start );
            if ( temp == pivot )
                return arg->desc->mt;
            else
                return temp;
            break;

        case DPLASMA_QR_KILLED_BY_DISTTREE:
            if ( start == pivot && (pivot + 1 >= arg->desc->mt) )
                return arg->desc->mt;
            else
                myassert( start != pivot );
            temp = arg->hlvl->prevpiv(arg->hlvl, pivot, k, start );
            if ( temp == pivot )
                return arg->desc->mt;
            else
                return temp;
            break;

        default:
            return arg->desc->mt;
        }
};

/****************************************************
 *
 * Generate the permutation required for the round-robin on TS
 *
 ***************************************************/
/* static void dplasma_qr_genperm( qr_piv_t *qrpiv ) */
/* { */
/*     int m = qrpiv->desc->mt; */
/*     int n = qrpiv->desc->nt; */
/*     int minMN = min( m, n ); */
/*     int i, j, k; */
/*     int *perm; */

/*     qrpiv->perm = (int*)malloc( (m+1) * minMN * sizeof(int) ); */
/*     perm = qrpiv->perm; */
/*     for(k=0; k<minMN; k++) { */
/*         for( i=0; i<m+1; i++) { */
/*             perm[i] = i; */
/*         } */
/*         perm += m+1; */
/*     } */
/* } */



/* int dplasma_qr_getinvperm( const qr_piv_t *qrpiv, int k, int m ) */
/* { */
/*     int p  = qrpiv->p; */
/*     int pa = qrpiv->a * qrpiv->p; */
/*     int start = m / pa * pa; */
/*     int stop  = min( start + pa, qrpiv->desc->mt+1 ) - start; */
/*     int *perm = qrpiv->perm + (qrpiv->desc->mt+1)*k; /\* + start;*\/ */
/*     int i; */

/*     for ( i=0; i < qrpiv->desc->mt+1; i++ ) { */
/*         if( perm[i] == m ) */
/*             return i; */
/*     } */

/*    /\* We should never arrive here *\/ */
/*     myassert( 0 ); */
/* } */


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
    int low_mt, minMN, sq_p;
    qr_piv_t *qrpiv = (qr_piv_t*) malloc( sizeof(qr_piv_t) );
    (void)a; (void)domino; (void)tsrr;
 
    p = max( p, 1 );
    sq_p = (int)sqrt((double)p);
    
    assert( p == (sq_p * sq_p) );

    qrpiv->desc = A;
    qrpiv->a = 1;
    qrpiv->p = p;
    qrpiv->domino = 0;
    qrpiv->tsrr = 0;
    qrpiv->perm = NULL;

    qrpiv->llvl = (qr_subpiv_t*) malloc( sizeof(qr_subpiv_t) );
    qrpiv->hlvl = NULL;

    minMN = min(A->mt, A->nt);
    low_mt = A->mt;

    qrpiv->llvl->minMN  = minMN;
    qrpiv->llvl->ldd    = low_mt;
    qrpiv->llvl->a      = 1;
    qrpiv->llvl->p      = p;
    qrpiv->llvl->domino = 0;

    switch( type_llvl ) {
    case DPLASMA_FLAT_TREE :
    default:
        dplasma_low_flat_init(qrpiv->llvl);
    }

    if ( p > 1 ) {
        qrpiv->hlvl = (qr_subpiv_t*) malloc( sizeof(qr_subpiv_t) );

        qrpiv->hlvl->minMN  = minMN; //a priori on ne l'utilise jamais ?
        qrpiv->hlvl->ldd    = A->mt;
        qrpiv->hlvl->a      = 1;
        qrpiv->hlvl->p      = sq_p;
        qrpiv->hlvl->domino = 0;

        switch( type_hlvl ) {
        case DPLASMA_FLAT_TREE :
        default:
            dplasma_high_flat_flat_init(qrpiv->hlvl);
        }
    }

    /*dplasma_qr_genperm( qrpiv );*/
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
