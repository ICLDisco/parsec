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
 *       For all subdiagonal "macro-tiles", the line reduced is always the first.
 *       For all diagonal "macro-tiles", the factorization performed
 *       is identical to the one performed by PLASMA_zgeqrf.
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
#include <math.h>
#include <plasma.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"
#include "dplasma_qr_pivgen.h"

#ifndef min
#define min(__a, __b) ( ( (__a) < (__b) ) ? (__a) : (__b) )
#endif

#ifndef max
#define max(__a, __b) ( ( (__a) > (__b) ) ? (__a) : (__b) )
#endif

#define PIV(_ipiv, _mt, _i, _k) (_piv)[(_k)*(_mt)+(_i)]
#define PRINT_PIVGEN 1

#ifdef PRINT_PIVGEN
#define myassert( test ) if ( ! (test) ) return -1;
#else
#define myassert assert
#endif

/*
 * Common functions
 */
int dplasma_qr_getnbgeqrf( const int a, const int p, const int domino, const int k, const int gmt );
int dplasma_qr_getm(       const int a, const int p, const int domino, const int k, const int i   );
int dplasma_qr_geti(       const int a, const int p, const int domino, const int k, const int m   );
int dplasma_qr_gettype(    const int a, const int p, const int domino, const int k, const int m   );

/*
 * Debug
 */
int  dplasma_qr_check(         tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_type(    tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_pivot(   tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_nbgeqrt( tiled_matrix_desc_t *A, qr_piv_t *qrpiv );
void dplasma_qr_print_next_k(  tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k );
void dplasma_qr_print_prev_k(  tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k );
void dplasma_qr_print_geqrt_k( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k );

/*
 * Subtree for low-level
 */
static void dplasma_low_flat_init(     qr_subpiv_t *arg);
static void dplasma_low_greedy_init(   qr_subpiv_t *arg, const int minMN);
static void dplasma_low_binary_init(   qr_subpiv_t *arg);
static void dplasma_low_fibonacci_init(qr_subpiv_t *arg, const int minMN);


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
int dplasma_qr_getnbgeqrf( const int a, const int p, const int domino, const int k, const int gmt ) {
    int pa = p * a;
    int nb_1, nb_2, nb_3;
    int nb_11, nb_12;

    /* Number of tasks of type 3 */
    nb_3 = p;
    
    /* Number of tasks of type 2 */
    if ( domino ) {
      nb_2 = k * (p - 1);

      /* First multiple of p*a under the diagonal of step k */
      nb_11 = ( ( (p * (k+1)) + pa - 1 ) / pa ) * pa;
    }
    else {
      /* Number of extra tile of type 1 between the the tile of type 3 and the first of nb11 */
      nb_2 = nbextra1_formula;

      /* First multiple of p*a under the diagonal of step 1 */
      nb_11 = ( (k + p + pa - 1 ) / pa ) * pa;
    }

    /* Last multiple of p*a lower than A->mt */
    nb_12 = ( gmt / pa ) * pa;

    /* Number of tasks of type 1 between nb_11 and nb_12 */
    nb_1 = (nb_12 - nb_11) / a;
    
    /* Add leftover */
    nb_1 += min( p, gmt - nb_12 );
    
    return min( nb_1 + nb_2 + nb_3, gmt - k);
}

/*
 * Extra parameter:
 *    i - indice of the geqrt in the continuous space
 * Return:
 *    The global indice m of the i th geqrt in the panel k
 */
int dplasma_qr_getm( const int a, const int p, const int domino, const int k, const int i)
{
    int pos1, j, pa = p * a;
    int nbextra1 = nbextra1_formula;
    int nb23 = p + (domino ? k*(p-1) : nbextra1 );

    /* Tile of type 2 or 3 or the 1 between the diagonal and the multiple after the diagonal */
    if ( i < nb23 )
        return k+i;
    /* Tile of type 1 */
    else {
        j = i - nb23;
        pa = p * a;
        if ( domino )
          pos1 = ( ( (p * (k+1)) + pa - 1 ) / pa ) * pa;
        else
          pos1 = ( ( (p + k    ) + pa - 1 ) / pa ) * pa;
        return pos1 + (j/p) * pa + j%p ;
    }
}

/*
 * Extra parameter:
 *    m - The global indice m of a geqrt in the panel k
 * Return:
 *    The index i of the geqrt in the panel k 
 */
int dplasma_qr_geti( const int a, const int p, const int domino, const int k, const int m)
{
    int pos1, j, pa = p * a;
    int nbextra1 = nbextra1_formula;
    int nb23 = p + ( domino ? k*(p-1) : nbextra1 );
    int end2 = p + ( domino ? k*p     : nbextra1 );

    /* Tile of type 2 or 3 or the 1 between the diagonal and the multiple after the diagonal */
    if ( m < end2 )
        return m-k;
    /* Tile of type 1 */
    else {
        if ( domino )
          pos1 = ( ( (p * (k+1)) + pa - 1 ) / pa ) * pa;
        else
          pos1 = ( ( (p + k    ) + pa - 1 ) / pa ) * pa;
        j = m - pos1;
        return nb23 + (j / pa) * p + j%pa ;
    }
}

/* 
 * Extra parameter:
 *      m - The global indice m of the row in the panel k
 * Return
 *     -1 - Error
 *      0 - if m is reduced thanks to a TS kernel
 *      1 - if m is reduced thanks to the low level tree
 *      2 - if m is reduced thanks to the bubble tree
 *      3 - if m is reduced in distributed
 */
int dplasma_qr_gettype( const int a, const int p, const int domino, const int k, const int m ) {
    myassert( m >= k );

    /* Element to be reduce in distributed */
    if (m < k + p) {
        return 3;
    }
    /* Element on the local diagonal */
    else if ( domino && m < p * (k+1) )
      return 2;
    /* Lower triangle of the matrix */
    else {
        if( (m / p) % a == 0 )
            return 1;
        else
            return 0;
    }
}

static int dplasma_qr_getinon0( const int a, const int p, const int domino, const int k, int i, int mt ) {

    int j;
    for(j=k; j<mt; j++) {
        if ( dplasma_qr_gettype( a, p, domino, k, j) != 0 )
            i--;
        if ( i == -1 )
            break;
    }
    return j;
}

/****************************************************
 *                 DPLASMA_LOW_FLAT_TREE
 ***************************************************/
static int dplasma_low_flat_currpiv(const qr_subpiv_t *arg, const int m, const int k) 
{ 
    (void)m;
    return k / arg->a;
};

static int dplasma_low_flat_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{ 
#ifdef FLAT_UP
    if ( ( p == (k/arg->a) ) && (start > (k/arg->a)+1 ) )
        return start-1;
    else
#else /* FLAT_DOWN */
    if ( start <= p )
        return arg->ldd;

    if ( arg->domino ) {
      if ( p == (k/arg->a) && ( arg->ldd - k/arg->a ) > 1 ) {
          if ( start == arg->ldd )
              return p+1;
          else if ( start < arg->ldd )
              return start+1;
      }
    } else {
        if ( (k <= p)  && (p < k + arg->p ) && ( arg->ldd - k/arg->a ) > 1 ) {
          if ( start == arg->ldd )
              return p+1;
          else if ( start < arg->ldd )
              return start+1;
      }
    }
#endif
    return arg->ldd;
};

static int dplasma_low_flat_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{ 
#ifdef FLAT_UP
    if ( p == (k/arg->a) && (start+1 < arg->ldd) )
      return start+1;
    else 
#else
      if ( p == (k/arg->a) && ( arg->ldd - k/arg->a ) > 1  ) { 
        if ( start == p )
            return arg->ldd - 1;
        else if ( start > p + 1 )
            return start-1;
    }
#endif
    return arg->ldd;
};

static void dplasma_low_flat_init(qr_subpiv_t *arg){
    arg->currpiv = dplasma_low_flat_currpiv;
    arg->nextpiv = dplasma_low_flat_nextpiv;
    arg->prevpiv = dplasma_low_flat_prevpiv;
    arg->ipiv = NULL;
};

/****************************************************
 *                 DPLASMA_LOW_BINARY_TREE
 ***************************************************/
static int dplasma_low_binary_currpiv(const qr_subpiv_t *arg, const int m, const int k) 
{ 
    int tmp1 = m - (k / arg->a);
    int tmp2 = 1;
    (void)arg;

    if ( tmp1 == 0) 
        return 0;
    while( (tmp1 != 0 ) && (tmp1 % 2 == 0) ) {
        tmp1 = tmp1 >> 1;
        tmp2 = tmp2 << 1;
    }        
    assert( m - tmp2 >= (k / arg->a) );
    return m - tmp2;
};

static int dplasma_low_binary_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{ 
    int tmpp, bit;
    int lk = (k / arg->a);

    myassert( (start == arg->ldd) || (dplasma_low_binary_currpiv( arg, start, k ) == p) );

    if ( start <= p )
        return arg->ldd;
    
    int offset = p-lk;
    bit = 0;
    if (start != arg->ldd) {
        while( ( (start-lk) & (1 << bit ) ) == 0 )
            bit++;
        bit++;
    }
    
    tmpp = offset | (1 << bit);
    if ( ( tmpp != offset ) && ( tmpp+lk < arg->ldd ) )
        return tmpp + lk;
    else
        return arg->ldd;
};

static int dplasma_low_binary_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{ 
    int lk = (k / arg->a);
    myassert( start >= p && ( start == p || dplasma_low_binary_currpiv( arg, start, k ) == p) );

    int offset = p-lk;
    if ( (start == p) && ( offset%2 == 0 ) ) {
        int i, bit, tmp;
        if ((p-lk) == 0)
            bit = (int)( log( (double)(arg->ldd - lk) ) / log( 2. ) );
        else {
            bit = 0;
            while( (offset & (1 << bit )) == 0 )
                bit++;
        }
        for( i=bit; i>-1; i--){
            tmp = offset | (1 << i);
            if ( ( offset != tmp ) && ( tmp+lk < arg->ldd ) )
                return tmp+lk;
        }                
        return arg->ldd;
    }

    if ( (start - p) > 1 )
        return p + ( (start-p) >> 1 );
    else {
        return arg->ldd;
    }
};

static void dplasma_low_binary_init(qr_subpiv_t *arg){
    arg->currpiv = dplasma_low_binary_currpiv;
    arg->nextpiv = dplasma_low_binary_nextpiv;
    arg->prevpiv = dplasma_low_binary_prevpiv;
    arg->ipiv = NULL;
};

/****************************************************
 *          DPLASMA_LOW_FIBONACCI_TREE
 ***************************************************/
/* Return the pivot to use for the row m at step k */
inline static int dplasma_low_fibonacci_currpiv( const qr_subpiv_t *qrpiv, const int m, const int k ) {
    return (qrpiv->ipiv)[ (k/qrpiv->a) * (qrpiv->ldd) + m ];
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int dplasma_low_fibonacci_prevpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    for( i=start+1; i<(qrpiv->ldd); i++ )
        if ( (qrpiv->ipiv)[i +  (k/qrpiv->a) * (qrpiv->ldd)] == p )
            return i;
    return i;
 }

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int dplasma_low_fibonacci_nextpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    for( i=start-1; i>(k/qrpiv->a); i-- )
        if ( (qrpiv->ipiv)[i + (k/qrpiv->a) * (qrpiv->ldd)] == p )
            return i;
    return (qrpiv->ldd);
}

static void dplasma_low_fibonacci_init(qr_subpiv_t *arg, int minMN){
    int *ipiv;
    int mt, a, p, domino;

    arg->currpiv = dplasma_low_fibonacci_currpiv;
    arg->nextpiv = dplasma_low_fibonacci_nextpiv;
    arg->prevpiv = dplasma_low_fibonacci_prevpiv;
    
    mt = arg->ldd;
    a = arg->a;
    p = arg->p;
    domino = arg->domino;

    arg->ipiv = (int*)malloc( mt * minMN * sizeof(int) );
    ipiv = arg->ipiv;
    memset(ipiv, 0, mt*minMN*sizeof(int));
   
    /*
     * Fibonacci of order 1:
     *    u_(n+1) = u_(n) + 1
     */
    {
        int f1, k, m;

        /* Fill in the first column */
        f1 = 1;
        for (m=1; m < mt; ) {
            for (k=0; (k < f1) && (m < mt); k++, m++) {
                ipiv[m] = m - f1;
            }
            f1++;
        }

        for( k=1; k<minMN; k++) {
            for(m=k+1; m < mt; m++) {
                ipiv[ k * mt + m ] = ipiv[ (k-1) * mt + m - 1 ] + 1;
            }
        }
    }
};

/****************************************************
 *          DPLASMA_LOW_GREEDY_TREE
 ***************************************************/
/* Return the pivot to use for the row m at step k */
inline static int dplasma_low_greedy_currpiv( const qr_subpiv_t *qrpiv, const int m, const int k ) {
    return (qrpiv->ipiv)[ k * (qrpiv->ldd) + m ];
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int dplasma_low_greedy_prevpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    for( i=start+1; i<(qrpiv->ldd); i++ )
        if ( (qrpiv->ipiv)[i +  k * (qrpiv->ldd)] == p )
            return i;
    return i;
 }

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int dplasma_low_greedy_nextpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    for( i=start-1; i>(k/qrpiv->a); i-- )
        if ( (qrpiv->ipiv)[i + k* (qrpiv->ldd)] == p )
            return i;
    return (qrpiv->ldd);
}

static void dplasma_low_greedy_init(qr_subpiv_t *arg, int minMN){
    int *ipiv;
    int mt, a, p, domino;
    
    arg->currpiv = dplasma_low_greedy_currpiv;
    arg->nextpiv = dplasma_low_greedy_nextpiv;
    arg->prevpiv = dplasma_low_greedy_prevpiv;

    mt = arg->ldd;
    a = arg->a;
    p = arg->p;
    domino = arg->domino;

    minMN = min( minMN, mt*a );
    arg->ipiv = (int*)malloc( mt * minMN * sizeof(int) );
    ipiv = arg->ipiv;
    memset(ipiv, 0, mt*minMN*sizeof(int));
  
    {
        int j, k, height, start, end, firstk = 0;
        int *nT = (int*)malloc(minMN*sizeof(int));
        int *nZ = (int*)malloc(minMN*sizeof(int));
        memset( nT, 0, minMN*sizeof(int));
        memset( nZ, 0, minMN*sizeof(int));
        nT[0] = mt;
        
        k = 0;
        while ( (!( ( nT[minMN-1] == mt - ( (minMN - 1) / a ) ) &&
                     ( nZ[minMN-1]+1 == nT[minMN-1] ) ) )
                && ( firstk < minMN ) ) {
            height = (nT[k] - nZ[k]) / 2;
            if ( height == 0 ) {
                while ( (firstk < minMN) &&
                        ( nT[firstk] == mt - ( firstk / a ) ) &&
                        ( nZ[firstk]+1 == nT[firstk] ) ) {
                    if (  (( firstk % a) != a-1 )
                          && ( firstk < minMN-1 ) )
                        nT[firstk+1]++;
                    firstk++;
                }
                k = firstk;
                continue;
            }
             
            if (k < minMN-1) nT[k+1] += height;
            start = mt - nZ[k] - 1;
            end = start - height;
            nZ[k] += height;
            
            for( j=start; j > end; j-- ) {
                ipiv[ k*mt + j ] = (j - height);
            }

            k++;
            if (k > minMN-1) k = firstk;
        }
        
        free(nT);
        free(nZ);
    }
#if 0
    {
        int m, k;
        for(m=0; m<mt; m++) {
            printf("%4d | ", m);              
            for (k=0; k<minMN; k++) {
                printf( "%3d ", ipiv[k*mt + m] );
            }
            printf("\n");
        }
    }
#endif
};

/****************************************************
 *                 DPLASMA_HIGH_FLAT_TREE
 ***************************************************/
static int dplasma_high_flat_currpiv(const qr_subpiv_t *arg, const int m, const int k) 
{ 
    (void)arg;
    (void)m;
    return k;
};

static int dplasma_high_flat_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{ 
    if ( p == k && arg->ldd > 1 ) {
        if ( start == arg->ldd )
            return p+1;
        else if ( start < arg->ldd && (start-k < arg->a-1) )
            return start+1;
    }
    return arg->ldd;
};

static int dplasma_high_flat_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{ 
    assert( arg->a > 1 );
    if ( p == k && arg->ldd > 1 ) { 
        if ( start == p && p != arg->ldd-1 )
            return min( p + arg->a - 1, arg->ldd - 1 );
        else if ( start > p + 1 && (start-k < arg->a))
            return start-1;
    }
    return arg->ldd;
};

static void dplasma_high_flat_init(qr_subpiv_t *arg){
    arg->currpiv = dplasma_high_flat_currpiv;
    arg->nextpiv = dplasma_high_flat_nextpiv;
    arg->prevpiv = dplasma_high_flat_prevpiv;
    arg->ipiv = NULL;
};

/****************************************************
 *                 DPLASMA_HIGH_GREEDY_TREE
 ***************************************************/
static int dplasma_high_greedy_currpiv(const qr_subpiv_t *arg, const int m, const int k) 
{ 
    myassert( m >= k && m < k+arg->a );
    return (arg->ipiv)[ k * (arg->a) + (m - k) ];
};

static int dplasma_high_greedy_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{ 
    int i;
    myassert( (start >= k && start < k+arg->a) || start == arg->ldd );
    for( i=min(start-1, k+arg->a-1); i > k; i-- )
        if ( (arg->ipiv)[i-k + k* (arg->a)] == p )
            return i;
    return (arg->ldd);
};

static int dplasma_high_greedy_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{ 
    int i;
    myassert( (start >= k && start < k+arg->a) || start == p );
    for( i=start-k+1; i<arg->a; i++ )
        if ( (arg->ipiv)[i +  k * (arg->a)] == p )
            return k+i;
    return arg->ldd;
};

static void dplasma_high_greedy_init(qr_subpiv_t *arg, int minMN){
    int *ipiv;
    int mt, a, p, domino;

    arg->currpiv = dplasma_high_greedy_currpiv;
    arg->nextpiv = dplasma_high_greedy_nextpiv;
    arg->prevpiv = dplasma_high_greedy_prevpiv;

    mt = arg->ldd;
    a = arg->a;
    p = arg->p;
    domino = arg->domino;

    arg->ipiv = (int*)malloc( a * minMN * sizeof(int) );
    ipiv = arg->ipiv;
    memset(ipiv, 0, a*minMN*sizeof(int));
  
    {
        int j, k, height, start, end, firstk = 0;
        int *nT = (int*)malloc(minMN*sizeof(int));
        int *nZ = (int*)malloc(minMN*sizeof(int));
        memset( nT, 0, minMN*sizeof(int));
        memset( nZ, 0, minMN*sizeof(int));

        nT[0] = mt;
        nZ[0] = max( mt - a, 0 );
        for(k=1; k<minMN; k++) {
            height = max(mt-k-a, 0);
            nT[k] = height;
            nZ[k] = height;
        }
        
        k = 0;
        while ( (!( ( nT[minMN-1] == mt - (minMN - 1) ) &&
                    ( nZ[minMN-1]+1 == nT[minMN-1] ) ) )
                && ( firstk < minMN ) ) {
            height = (nT[k] - nZ[k]) / 2;
            if ( height == 0 ) {
                while ( (firstk < minMN) &&
                        ( nT[firstk] == mt - firstk ) &&
                        ( nZ[firstk]+1 == nT[firstk] ) ) {
                    firstk++;
                }
                k = firstk;
                continue;
            }
             
            start = mt - nZ[k] - 1;
            end = start - height;
            nZ[k] += height;
            if (k < minMN-1) nT[k+1] = nZ[k];
            
            for( j=start; j > end; j-- ) {
                ipiv[ k*a + j-k ] = (j - height);
            }

            k++;
            if (k > minMN-1) k = firstk;
        }
        
        free(nT);
        free(nZ);
    }
};

/****************************************************
 *       DPLASMA_LOW_TREE / DPLASMA_HIGH_TREE
 ***************************************************/
int dplasma_qr_currpiv(const qr_piv_t *arg, const int m, const int k) 
{ 
    int tmp;
    int a    = arg->a;
    int p    = arg->p;
    int domino = arg->domino;
    int lm   = m / p; /* Local index in the distribution over p domains */
    int rank = m % p; /* Staring index in this distribution             */

    /* TS level common to every case */
    switch( dplasma_qr_gettype( a, p, domino, k, m ) ) 
        {
        case 0:
            return ( (lm / a) == (k / a) ) ? k*p+rank : (lm / a) * a * p + rank;
            break;
        case 1:
            tmp = arg->llvl->currpiv(arg->llvl, lm / a, k);
            return ( (k / a) == tmp ) ? k * p + rank : tmp * a * p + rank;
            break;
        case 2:
            return m - p;
            break;
        case 3:
            if ( arg->hlvl != NULL )
                return arg->hlvl->currpiv(arg->hlvl, m, k);
        default:
            return arg->desc->mt;
        }
};

int dplasma_qr_nextpiv(const qr_piv_t *arg, const int pivot, const int k, int start)
{ 
    int tmp, ls, lp, nextp;
    int a    = arg->a;
    int p    = arg->p;
    int domino = arg->domino;
    int lpivot = pivot / p; /* Local index in the distribution over p domains */
    int rpivot = pivot % p; /* Staring index in this distribution             */
    /* Local index in the distribution over p domains */
    int lstart = ( start == arg->desc->mt ) ? arg->llvl->ldd * a : start / p;

    myassert( start > pivot && pivot >= k );
    myassert( start == arg->desc->mt || pivot == dplasma_qr_currpiv( arg, start, k ) );
            
    /* TS level common to every case */
    ls = (start < arg->desc->mt) ? dplasma_qr_gettype( a, p, domino, k, start ) : -1;
    lp = dplasma_qr_gettype( a, p, domino, k, pivot );

    switch( ls ) 
        {
        case -1:

            if ( lp == 0 ) {
                myassert( start == arg->desc->mt );
                return arg->desc->mt;
            }
            
        case 0:

            /* If the tile is over the diagonal of step k, skip directly to type 2 */
            if ( lpivot < k )
                goto next_2;
                    
            if ( start == arg->desc->mt )
                nextp = pivot + p;
            else
                nextp = start + p;

            if ( ( nextp < arg->desc->mt ) && 
                 ( nextp < pivot + a*p ) &&
                 ( (nextp/p)%a != 0 ) )
                return nextp;
            
            /* /\* First query / Check for use in TS *\/ */
            /* if ( start == arg->desc->mt ) { */
            /*     tmp = lpivot + a - 1 - lpivot%a; */
            /*     nextp = tmp * p + rpivot; */
                
            /*     while( pivot < nextp && nextp >= arg->desc->mt )  */
            /*         nextp -= p; */
            /* } else { */
            /*     nextp = (lstart - 1) * p + rpivot; */
            /* }                 */
            /* if ( pivot < nextp && nextp < arg->desc->mt )  */
            /*     return nextp;  */

            /* no next of type 0, we reset start to search the next 1 */
            start = arg->desc->mt;
            lstart = arg->llvl->ldd * a;
                
        case 1:

            /* If the tile is over the diagonal of step k, skip directly to type 2 */
            if ( lpivot < k )
                goto next_2;
                    
            /* Get the next pivot for the low level tree */
            tmp = arg->llvl->nextpiv(arg->llvl, lpivot / a, k, lstart / a );

            if ( (tmp * a * p + rpivot >= arg->desc->mt)
                 && (tmp == arg->llvl->ldd-1) )
                tmp = arg->llvl->nextpiv(arg->llvl, lpivot / a, k, tmp);
            
            if ( tmp != arg->llvl->ldd )
                return tmp * a * p + rpivot;

        next_2:
            /* no next of type 1, we reset start to search the next 2 */
            start = arg->desc->mt;
            lstart = arg->llvl->ldd * a;
            
        case 2:

            if ( lp < 2 ) {
                return arg->desc->mt;
            }

            /* Type 2 are killed only once if they are strictly in the band */
            if ( (start == arg->desc->mt) && 
                 (lpivot < k)             &&
                 (pivot+p < arg->desc->mt) ) {
                return pivot+p;
            }

            /* no next of type 2, we reset start to search the next 3 */
            start = arg->desc->mt;
            lstart = arg->llvl->ldd * a;

        case 3:

            if ( lp < 3 ) {
                return arg->desc->mt;
            }

            if( arg->hlvl != NULL ) {
                tmp = arg->hlvl->nextpiv( arg->hlvl, pivot, k, start );
                if ( tmp != arg->desc->mt ) 
                    return tmp;
            }

        default:
            return arg->desc->mt;
        }
}

int dplasma_qr_prevpiv(const qr_piv_t *arg, const int pivot, const int k, int start)
{ 
    int tmp, ls, lp, nextp;
    int a = arg->a;
    int p = arg->p;
    int domino = arg->domino;
    int lpivot = pivot / p; /* Local index in the distribution over p domains */
    int rpivot = pivot % p; /* Staring index in this distribution             */
    int lstart = start / p; /* Local index in the distribution over p domains */

    myassert( start >= pivot && pivot >= k && start < arg->desc->mt );
    myassert( start == pivot || pivot == dplasma_qr_currpiv( arg, start, k ) );
            
    /* TS level common to every case */
    ls = dplasma_qr_gettype( a, p, domino, k, start );
    lp = dplasma_qr_gettype( a, p, domino, k, pivot );

    if ( lp == 0 )
      return arg->desc->mt;

    myassert( lp >= ls );
    switch( ls )
        {
        case 3:
            if( arg->hlvl != NULL ) {
                tmp = arg->hlvl->prevpiv( arg->hlvl, pivot, k, start );
                if ( tmp != arg->desc->mt )
                    return tmp;
            }

            start = pivot;
            lstart = pivot / p;

        case 2:
            if ( lpivot < k ) {
                
                if ( ( start == pivot ) &&
                     (start+p < arg->desc->mt ) )
                    return start+p;
                
                if ( lp > 1 )
                    return  arg->desc->mt;
            }

            start = pivot;
            lstart = pivot / p;
            
            /* If it is the 'local' diagonal block, we go to 1 */

        case 1:
            if ( lpivot < k )
                return  arg->desc->mt;
                 
            tmp = arg->llvl->prevpiv(arg->llvl, lpivot / a, k, lstart / a);

            if ( (tmp * a * p + rpivot >= arg->desc->mt)
                 && (tmp == arg->llvl->ldd-1) )
                tmp = arg->llvl->prevpiv(arg->llvl, lpivot / a, k, tmp);
                
            if ( tmp != arg->llvl->ldd )
                return tmp * a * p + rpivot;
            
            start = pivot;
            
        case 0:
            /* Search for predecessor in TS tree */
            /* if ( ( start+p < arg->desc->mt ) &&  */
            /*      ( (((start+p) / p) % a) != 0 ) ) */
            /*     return start + p; */
            
            if ( start == pivot ) {
                tmp = lpivot + a - 1 - lpivot%a;
                nextp = tmp * p + rpivot;
                
                while( pivot < nextp && nextp >= arg->desc->mt ) 
                    nextp -= p;
            } else {
                nextp = start - p; /*(lstart - 1) * p + rpivot;*/
            }                
            assert(nextp < arg->desc->mt);
            if ( pivot < nextp ) 
                return nextp; 

        default:
            return arg->desc->mt;
        }
};

/****************************************************
 ***************************************************/
/* #define ENDCHECK( test, ret )                   \ */
/*     if ( !test )                                \ */
/*         exit( -1 ); */

#define ENDCHECK( test, ret )                   \
    if ( !test )                                \
        return ret;

int dplasma_qr_check( tiled_matrix_desc_t *A, qr_piv_t *qrpiv)
{
    int minMN = min(A->mt, A->nt );
    int i, m, k, nb;
    int check;

    int a = qrpiv->a;
    int p = qrpiv->p;
    int domino = qrpiv->domino;

    /* 
     * Check Formula for NB geqrt 
     */
    {
        /* dplasma_qr_print_type( A, qrpiv ); */
        /* dplasma_qr_print_nbgeqrt( A, qrpiv ); */
        check = 1;
        for (k=0; k<minMN; k++) {
            nb = 0;
            for (m=k; m < A->mt; m++) {
              if ( dplasma_qr_gettype(qrpiv->a, qrpiv->p, domino, k, m) > 0 )
                    nb++;
            }

            if ( nb != dplasma_qr_getnbgeqrf( a, p, domino, k, A->mt) ) {
                check = 0;
                printf(" ----------------------------------------------------\n"
                       "  - a = %d, p = %d, M = %d, N = %d\n"
                       "     Check number of geqrt:\n"
                       "       For k=%d => return %d instead of %d",
                       a, p, A->mt, A->nt, k, dplasma_qr_getnbgeqrf( a, p, domino, k, A->mt), nb );
            }
        }
        
        ENDCHECK( check, 1 );
    }

#if 0
    /* 
     * Check indices of geqrt 
     */
    {
        int prevm = -1;
        check = 1;
        for (k=0; k<minMN; k++) {
            /* dplasma_qr_print_geqrt_k( A, qrpiv, k ); */
            nb = dplasma_qr_getnbgeqrf( a, p, domino, k, A->mt );
            prevm = -1;
            for (i=0; i < nb; i++) {

                m = dplasma_qr_getm( a, p, domino, k, i );

                /* tile before the diagonal are factorized and 
                 * the m is a growing list
                 */
                if ( ( m < k ) || ( m < prevm ) ) {
                    check = 0;
                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Check indices of geqrt:\n"
                           "        getm( k=%d, i=%d ) => m = %d", 
                           a, p, A->mt, A->nt, k, i, m);
                } else if ( m != dplasma_qr_getinon0( a, p, domino, k, i, A->mt ) ) {
                    check = 0;
                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Check indices of geqrt:\n"
                           "        getm( k=%d, i=%d ) => m = %d but should be %d", 
                           a, p, A->mt, A->nt, k, i, m, dplasma_qr_getinon0( a, p, domino, k, i, A->mt));
                } else if ( i != dplasma_qr_geti( a, p, domino, k, m) ) {
                    check = 0;
                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Check indices of geqrt:\n"
                           "        getm( k=%d, i=%d ) => m = %d && geti( k=%d, m=%d ) => i = %d\n", 
                           a, p, A->mt, A->nt, 
                           k, i, m, k, m, dplasma_qr_geti( a, p, domino, k, m));
                }
                prevm = m;
            }
        }
        ENDCHECK( check, 2 );
    }

    /* 
     * Check number of exit in next
     */
    {
        int s;
        check = 1;
 
        for (k=0; k<minMN; k++) {
            for(m=k; m<A->mt; m++) {
                nb = 0;
                for(s=A->mt; s>k; s--) {
                    if ( dplasma_qr_nextpiv(qrpiv, m, k, s) == A->mt )
                        nb++;
                }
                if ( nb > 1 ) {
                    dplasma_qr_print_next_k( A, qrpiv, k);
                    dplasma_qr_print_prev_k( A, qrpiv, k);

                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Next of line %d for step %d contains more than one exit:\n",
                           a, p, A->mt, A->nt, 
                           m, k);
                    check = 0;
                    return 3;
                }
                else if ( nb == 0 ) {
                    dplasma_qr_print_next_k( A, qrpiv, k);
                    dplasma_qr_print_prev_k( A, qrpiv, k);

                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Next of line %d for step %d needs one exit:\n",
                           a, p, A->mt, A->nt, 
                           m, k);
                    check = 0;
                    return 3;
                }
            }
        }
        ENDCHECK( check, 3 );
    }

    /* 
     * Check number of exit in prev
     */
    {
        int s;
        check = 1;
 
        for (k=0; k<minMN; k++) {
            for(m=k; m<A->mt; m++) {
                nb = 0;
                for(s=k; s<A->mt; s++) {
                    if ( dplasma_qr_prevpiv(qrpiv, m, k, s) == A->mt )
                        nb++;
                }
                if ( nb > 1 ) {
                    dplasma_qr_print_next_k( A, qrpiv, k);
                    dplasma_qr_print_prev_k( A, qrpiv, k);

                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Prev of line %d for step %d contains more than one exit:\n",
                           a, p, A->mt, A->nt, 
                           m, k);
                    check = 0;
                    return 3;
                }
                else if ( nb == 0 ) {
                    dplasma_qr_print_next_k( A, qrpiv, k);
                    dplasma_qr_print_prev_k( A, qrpiv, k);

                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Prev of line %d for step %d needs one exit:\n",
                           a, p, A->mt, A->nt, 
                           m, k);
                    check = 0;
                    return 3;
                }
            }
        }
        ENDCHECK( check, 3 );
    }

    /* 
     * Check next/prev
     */
    {
        int start, next, prev;
        check = 1;
 
        for (k=0; k<minMN; k++) {
            /* dplasma_qr_print_next_k( A, qrpiv, k); */
            /* dplasma_qr_print_prev_k( A, qrpiv, k); */

            start = A->mt;
            for(m=k; m<A->mt; m++) {

                do {
                    next = dplasma_qr_nextpiv(qrpiv, m, k, start);
                    if ( next == A->mt ) 
                        prev = dplasma_qr_prevpiv(qrpiv, m, k, m);
                    else 
                        prev = dplasma_qr_prevpiv(qrpiv, m, k, next);
                    
                    if ( start != prev ) {
                        dplasma_qr_print_next_k( A, qrpiv, k);
                        dplasma_qr_print_prev_k( A, qrpiv, k);

                        printf(" ----------------------------------------------------\n"
                               "  - a = %d, p = %d, M = %d, N = %d\n"
                               "     Check next/prev:\n"
                               "       next( m=%d, k=%d, start=%d ) => %d && prev( m=%d, k=%d, start=%d ) => %d\n ( %d != %d )", 
                               a, p, A->mt, A->nt, 
                               m, k, start, next, m, k, next, prev, start, prev);
                        check = 0;
                        return 3;
                    }
                    start = next;
                } while ( start != A->mt );
            }
        }
        ENDCHECK( check, 3 );
    }
#endif
    return 0;
}


qr_piv_t *dplasma_pivgen_init( tiled_matrix_desc_t *A, int type_llvl, int type_hlvl, int a, int p, int domino )
{
    int low_mt, minMN;
    qr_piv_t *qrpiv = (qr_piv_t*) malloc( sizeof(qr_piv_t) );

    a = max( a, 1 );
    p = max( p, 1 );
    domino = domino ? 1 : 0;

    qrpiv->desc = A;
    qrpiv->a = a;
    qrpiv->p = p;
    qrpiv->domino = domino;

    qrpiv->llvl = (qr_subpiv_t*) malloc( sizeof(qr_subpiv_t) );
    qrpiv->hlvl = NULL;

    minMN = min(A->mt, A->nt);
    low_mt = (A->mt + p * a - 1) / ( p * a );

    qrpiv->llvl->ldd    = low_mt;
    qrpiv->llvl->a      = a;
    qrpiv->llvl->p      = p;
    qrpiv->llvl->domino = domino;

    switch( type_llvl ) {
    case DPLASMA_GREEDY_TREE :
        dplasma_low_greedy_init(qrpiv->llvl, minMN);
        break;
    case DPLASMA_FIBONACCI_TREE :
        dplasma_low_fibonacci_init(qrpiv->llvl, minMN);
        break;
    case DPLASMA_BINARY_TREE :
        dplasma_low_binary_init(qrpiv->llvl);
        break;
    case DPLASMA_FLAT_TREE :
    default:
        dplasma_low_flat_init(qrpiv->llvl);
    }

    if ( p > 1 ) {
        qrpiv->hlvl = (qr_subpiv_t*) malloc( sizeof(qr_subpiv_t) );

        qrpiv->hlvl->ldd    = A->mt;
        qrpiv->hlvl->a      = p;
        qrpiv->hlvl->p      = p;
        qrpiv->hlvl->domino = domino;

        switch( type_hlvl ) {
        case DPLASMA_GREEDY_TREE :
            dplasma_high_greedy_init(qrpiv->hlvl, minMN);
            break;
        /* case DPLASMA_FIBONACCI_TREE : */
        /*     printf("High level: Fibonacci\n"); */
        /*     dplasma_fibonacci_init(qrpiv->llvl, high_mt, min(A->mt, A->nt), a); */
        /*     break; */
        /* case DPLASMA_BINARY_TREE : */
        /*     printf("High level: Binary\n"); */
        /*     dplasma_binary_init(qrpiv->llvl, high_mt, a); */
        /*     break; */
        case DPLASMA_FLAT_TREE :
        default:
            dplasma_high_flat_init(qrpiv->hlvl);
        }
    } 

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
}

void dplasma_qr_print_type( tiled_matrix_desc_t *A, qr_piv_t *qrpiv )
{
    int minMN = min(A->mt, A->nt );
    int m, k;
    int lm = 0;
    int lmg = 0;
    int rank = 0;
    int domino = qrpiv->domino;

    printf("\n------------ Localization = Type of pivot --------------\n");
    for(m=0; m<A->mt; m++) {
        printf("%4d | ", m);              
        for (k=0; k<min(minMN, m+1); k++) {
            printf( "%3d ", dplasma_qr_gettype( qrpiv->a, qrpiv->p, domino, k, m ) );
        }
        for (k=min(minMN, m+1); k<minMN; k++) {
            printf( "    " );
        }
      
        printf("    ");
        printf("%2d,%4d | ", rank, lmg);
        for (k=0; k<min(minMN, lmg+1); k++) {
            printf( "%3d ", dplasma_qr_gettype( qrpiv->a, qrpiv->p, domino, k, lmg) );
        }
        for (k=min(minMN, lmg+1); k<minMN; k++) {
            printf( "    " );
        }
        lm++; lmg+=qrpiv->p;
        if ( lmg >= A->mt ) {
            rank++;
            lmg = rank;
            lm = 0;
        } 
        printf("\n");
    }
}

void dplasma_qr_print_pivot( tiled_matrix_desc_t *A, qr_piv_t *qrpiv )
{
    int minMN = min(A->mt, A->nt );
    int m, k;
    int lm = 0;
    int lmg = 0;
    int rank = 0;
    printf("\n------------ Current Pivot--------------\n");
    for(m=0; m<A->mt; m++) {
        printf("%4d | ", m);              
        for (k=0; k<min(minMN, m+1); k++) {
            printf( "%3d ", dplasma_qr_currpiv(qrpiv, m, k) );
        }
        for (k=min(minMN, m+1); k<minMN; k++) {
            printf( "    " );
        }
            
        printf("    ");
        printf("%2d,%4d | ", rank, lmg);
        for (k=0; k<min(minMN, lmg+1); k++) {
            printf( "%3d ", dplasma_qr_currpiv(qrpiv, lmg, k) );
        }
        for (k=min(minMN, lmg+1); k<minMN; k++) {
            printf( "    " );
        }
        lm++; lmg+=qrpiv->p;
        if ( lmg >= A->mt ) {
            rank++;
            lmg = rank;
            lm = 0;
        } 
        printf("\n");
    }
}

void dplasma_qr_print_next_k( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k )
{
    int m, s;
    printf("\n------------ Next (k = %d)--------------\n", k);

    printf( "       " );
    for(s=A->mt; s>0; s--)
        printf( "%4d  ", s );
    printf( "\n" );

    for(m=0; m<A->mt; m++) {
        printf("%4d | ", m);              
        for(s=A->mt; s>0; s--) {
            printf( "%4d  ", dplasma_qr_nextpiv(qrpiv, m, k, s) );
        }
        printf("\n");
    }
}

void dplasma_qr_print_prev_k( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k )
{
    int m, s;
    printf("\n------------ Prev (k = %d)--------------\n", k);

    printf( "       " );
    for(s=A->mt; s>-1; s--)
        printf( "%4d  ", s );
    printf( "\n" );

    for(m=0; m<A->mt; m++) {
        printf("%4d | ", m);              
        for(s=A->mt; s>-1; s--) {
            printf( "%4d  ", dplasma_qr_prevpiv(qrpiv, m, k, s) );
        }
        printf("\n");
    }
}


void dplasma_qr_print_nbgeqrt( tiled_matrix_desc_t *A, qr_piv_t *qrpiv )
{
    int minMN = min(A->mt, A->nt );
    int m, k, nb;
    int domino = qrpiv->domino;

    printf("\n------------ Nb GEQRT per k --------------\n");
    printf(" k      : ");
    for (k=0; k<minMN; k++) {
        printf( "%3d ", k );
    }
    printf( "\n" );
    printf(" Compute: ");
    for (k=0; k<minMN; k++) {
        nb = 0;
        for (m=k; m < A->mt; m++) {
            if ( dplasma_qr_gettype(qrpiv->a, qrpiv->p, domino, k, m) > 0 )
                nb++;
        }
        printf( "%3d ", nb );
    }
    printf( "\n" );
    printf(" Formula: ");
    for (k=0; k<minMN; k++) {
        printf( "%3d ", dplasma_qr_getnbgeqrf( qrpiv->a, qrpiv->p, domino, k, A->mt) );
    }
    printf( "\n" );
}

void dplasma_qr_print_geqrt_k( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k )
{
    int i, m, nb;
    int domino = qrpiv->domino;

    printf("\n------------ Liste of geqrt for k = %d --------------\n", k);

    printf( "  m:");
    nb = dplasma_qr_getnbgeqrf( qrpiv->a, qrpiv->p, domino, k, A->mt );
    for (i=0; i < nb; i++) {
        m = dplasma_qr_getm( qrpiv->a, qrpiv->p, domino, k, i );
        if ( i == dplasma_qr_geti( qrpiv->a, qrpiv->p, domino, k, m) )
            printf( "%3d ", m );
        else
            printf( "x%2d ", dplasma_qr_geti( qrpiv->a, qrpiv->p, domino, k, m) );
    }
    printf( "\n" );
}


