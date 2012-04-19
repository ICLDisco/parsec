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

/*
 * Common functions
 */
int dplasma_qr_getnbgeqrf( const qr_piv_t *arg, const int k, const int gmt );
int dplasma_qr_getm(       const qr_piv_t *arg, const int k, const int i   );
int dplasma_qr_geti(       const qr_piv_t *arg, const int k, const int m   );
int dplasma_qr_gettype(    const qr_piv_t *arg, const int k, const int m   );
int dplasma_qr_getsize( const qr_piv_t *arg, const int k, const int i );
int dplasma_qr_nexttriangle(const qr_piv_t *arg, int p, const int k, int m);

static void dplasma_qr_genperm   (       qr_piv_t *qrpiv );
static int  dplasma_qr_getinvperm( const qr_piv_t *qrpiv, const int k, int m );

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
int dplasma_qr_getnbgeqrf( const qr_piv_t *arg, const int k, const int gmt ) {
    int a = arg->a;
    int p = arg->p;
    int domino = arg->domino;
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
int dplasma_qr_getm( const qr_piv_t *arg, const int k, const int i )
{
    int  a = arg->a;
    int  p = arg->p;
    int  domino = arg->domino;
    int *perm   = arg->perm + (arg->desc->mt+1) * k;

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
        return perm[ pos1 + (j/p) * pa + j%p ];
    }
}

/*
 * Extra parameter:
 *    m - The global indice m of a geqrt in the panel k
 * Return:
 *    The index i of the geqrt in the panel k
 */
int dplasma_qr_geti( const qr_piv_t *arg, const int k, int m )
{
    int a = arg->a;
    int p = arg->p;
    int domino = arg->domino;
    int lm = dplasma_qr_getinvperm( arg, k, m );

    int pos1, j, pa = p * a;
    int nbextra1 = nbextra1_formula;
    int nb23 = p + ( domino ? k*(p-1) : nbextra1 );
    int end2 = p + ( domino ? k*p     : k + nbextra1 );

    /* Tile of type 2 or 3 or the 1 between the diagonal and the multiple after the diagonal */
    if ( lm < end2 )
        return lm-k;
    /* Tile of type 1 */
    else {
        if ( domino )
          pos1 = ( ( (p * (k+1)) + pa - 1 ) / pa ) * pa;
        else
          pos1 = ( ( (p + k    ) + pa - 1 ) / pa ) * pa;
        j = lm - pos1;
        return nb23 + (j / pa) * p + j%pa;
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
int dplasma_qr_gettype( const qr_piv_t *arg, const int k, const int m ) {
    int a = arg->a;
    int p = arg->p;
    int domino = arg->domino;

    int lm = dplasma_qr_getinvperm( arg, k, m );
    myassert( lm >= k );

    /* Element to be reduce in distributed */
    if (lm < k + p) {
        return 3;
    }
    /* Element on the local diagonal */
    else if ( domino && lm < p * (k+1) )
      return 2;
    /* Lower triangle of the matrix */
    else {
        if( (lm / p) % a == 0 )
            return 1;
        else
            return 0;
    }
}

/*
 * Extra parameter:
 *      i - The index of the geqrt in the panel k
 * Return
 *     The size of domain in tiles
 *
 * Warning: It doesn't work with domino
 */
int dplasma_qr_getsize( const qr_piv_t *arg, const int k, const int i ) {
    int a = arg -> a;
    int nb_tile = arg->desc->mt - k;
    int q = nb_tile / a;
    int r = nb_tile - q*a;
    
    if (i<r)
        return a-1;
    else
        return a;
}


/****************************************************
 *                 DPLASMA_LOW_FLAT_TREE
 ***************************************************/
static int dplasma_low_flat_currpiv(const qr_subpiv_t *arg, const int m, const int k)
{
    (void)m;
    if ( arg->domino )
        return k / arg->a;
    else
        return (k + arg->p - 1 - m%(arg->p)) / arg->p / arg->a ;
};

static int dplasma_low_flat_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start_pa)
{
    int k_a = arg->domino ? k / arg->a :  (k + arg->p - 1 - p%(arg->p)) / arg->p / arg->a;
    int p_pa = (p / arg->p ) / arg->a;

#ifdef FLAT_UP
    if ( ( p_pa == k_a ) && (start_pa > k_a+1 ) )
        return start_pa-1;
    else
#else /* FLAT_DOWN */
    if ( start_pa <= p_pa )
        return arg->ldd;

    if ( p_pa == k_a && ( arg->ldd - k_a ) > 1 ) {
        if ( start_pa == arg->ldd )
            return p_pa+1;
        else if ( start_pa < arg->ldd )
            return start_pa+1;
    }
#endif
    return arg->ldd;
}

static int dplasma_low_flat_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start_pa)
{
    int k_a = arg->domino ? k / arg->a :  (k + arg->p - 1 - p%(arg->p)) / arg->p / arg->a;
    int p_pa = (p / arg->p ) / arg->a;

#ifdef FLAT_UP
    if ( p_pa == k_a && (start_pa+1 < arg->ldd) )
      return start_pa+1;
    else
#else
      if ( p_pa == k_a && ( arg->ldd - k_a ) > 1  ) {
        if ( start_pa == p_pa )
            return arg->ldd - 1;
        else if ( start_pa > p_pa + 1 )
            return start_pa-1;
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
    int k_a = arg->domino ? k / arg->a :  (k + arg->p - 1 - m%(arg->p)) / arg->p / arg->a;
    int m_pa = (m / arg->p ) / arg->a;

    int tmp1 = m_pa - k_a;
    int tmp2 = 1;
    (void)arg;

    if ( tmp1 == 0)
        return 0;
    while( (tmp1 != 0 ) && (tmp1 % 2 == 0) ) {
        tmp1 = tmp1 >> 1;
        tmp2 = tmp2 << 1;
    }
    assert( m_pa - tmp2 >= k_a );
    return m_pa - tmp2;
};

static int dplasma_low_binary_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start_pa)
{
    int k_a = arg->domino ? k / arg->a :  (k + arg->p - 1 - p%(arg->p)) / arg->p / arg->a;
    int p_pa = (p / arg->p ) / arg->a;

    int tmpp, bit;
    myassert( (start_pa == arg->ldd) || (dplasma_low_binary_currpiv( arg, start_pa*arg->a*arg->p, k ) == p_pa || !arg->domino) );

    if ( start_pa <= p_pa )
        return arg->ldd;

    int offset = p_pa-k_a;
    bit = 0;
    if (start_pa != arg->ldd) {
        while( ( (start_pa-k_a) & (1 << bit ) ) == 0 )
            bit++;
        bit++;
    }

    tmpp = offset | (1 << bit);
    if ( ( tmpp != offset ) && ( tmpp+k_a < arg->ldd ) )
        return tmpp + k_a;
    else
        return arg->ldd;
};

static int dplasma_low_binary_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start_pa)
{
    int k_a = arg->domino ? k / arg->a :  (k + arg->p - 1 - p%(arg->p)) / arg->p / arg->a;
    int p_pa = (p / arg->p ) / arg->a;
    int offset = p_pa - k_a;

    myassert( start_pa >= p_pa && ( start_pa == p_pa || !arg->domino ||
                                    dplasma_low_binary_currpiv( arg, start_pa*arg->a*arg->p, k ) == p_pa ) );

    if ( (start_pa == p_pa) && ( offset%2 == 0 ) ) {
        int i, bit, tmp;
        if ((p_pa - k_a) == 0)
            bit = (int)( log( (double)(arg->ldd - k_a) ) / log( 2. ) );
        else {
            bit = 0;
            while( (offset & (1 << bit )) == 0 )
                bit++;
        }
        for( i=bit; i>-1; i--){
            tmp = offset | (1 << i);
            if ( ( offset != tmp ) && ( tmp+k_a < arg->ldd ) )
                return tmp+k_a;
        }
        return arg->ldd;
    }

    if ( (start_pa - p_pa) > 1 )
        return p_pa + ( (start_pa - p_pa) >> 1 );
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
    int k_a = qrpiv->domino ? k / qrpiv->a :  (k + qrpiv->p - 1 - m%(qrpiv->p)) / qrpiv->p / qrpiv->a;
    return (qrpiv->ipiv)[ k_a * (qrpiv->ldd) + ( (m / qrpiv->p) / qrpiv->a ) ];
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int dplasma_low_fibonacci_prevpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start_pa ) {
    int i;
    int k_a = qrpiv->domino ? k / qrpiv->a :  (k + qrpiv->p - 1 - p%(qrpiv->p)) / qrpiv->p / qrpiv->a;
    int p_pa = (p / qrpiv->p ) / qrpiv->a;

    for( i=start_pa+1; i<(qrpiv->ldd); i++ )
        if ( (qrpiv->ipiv)[i +  k_a * (qrpiv->ldd)] == p_pa )
            return i;
    return i;
 }

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int dplasma_low_fibonacci_nextpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start_pa ) {
    int i;
    int k_a = qrpiv->domino ? k / qrpiv->a :  (k + qrpiv->p - 1 - p%(qrpiv->p)) / qrpiv->p / qrpiv->a;
    int p_pa = (p / qrpiv->p ) / qrpiv->a;

    for( i=start_pa-1; i>k_a; i-- )
        if ( (qrpiv->ipiv)[i + k_a * (qrpiv->ldd)] == p_pa )
            return i;
    return (qrpiv->ldd);
}

static void dplasma_low_fibonacci_init(qr_subpiv_t *arg, int minMN){
    int *ipiv;
    int mt;

    arg->currpiv = dplasma_low_fibonacci_currpiv;
    arg->nextpiv = dplasma_low_fibonacci_nextpiv;
    arg->prevpiv = dplasma_low_fibonacci_prevpiv;

    mt = arg->ldd;

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
    if (qrpiv->domino)
        return (qrpiv->ipiv)[ k * (qrpiv->ldd) + ( (m / qrpiv->p) / qrpiv->a ) ];
    else
        return (qrpiv->ipiv)[ ( (m%qrpiv->p) * qrpiv->minMN + k ) * (qrpiv->ldd)
                              + ( ( m  / qrpiv->p ) / qrpiv->a ) ];
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int dplasma_low_greedy_prevpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start_pa ) {
    int i;
    int p_pa = p / qrpiv->p / qrpiv->a;
    int *ipiv = qrpiv->domino ? qrpiv->ipiv : qrpiv->ipiv + p%qrpiv->p * qrpiv->minMN *qrpiv->ldd;

    for( i=start_pa+1; i<(qrpiv->ldd); i++ )
        if ( ipiv[i +  k * (qrpiv->ldd)] == p_pa )
            return i;
    return i;
 }

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int dplasma_low_greedy_nextpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start_pa ) {
    int i;
    int pa = qrpiv->p * qrpiv->a;
    int k_a = qrpiv->domino ? k / qrpiv->a :  (k + qrpiv->p - 1 - p%(qrpiv->p)) / qrpiv->p / qrpiv->a;
    int p_pa = p / pa;
    int *ipiv = qrpiv->domino ? qrpiv->ipiv : qrpiv->ipiv + p%qrpiv->p * qrpiv->minMN *qrpiv->ldd;

    for( i=start_pa-1; i> k_a; i-- )
        if ( ipiv[i + k * (qrpiv->ldd)] == p_pa )
            return i;

    return (qrpiv->ldd);
}

static void dplasma_low_greedy_init(qr_subpiv_t *arg, int minMN){
    int *ipiv;
    int mt, a, p, pa, domino;

    arg->currpiv = dplasma_low_greedy_currpiv;
    arg->nextpiv = dplasma_low_greedy_nextpiv;
    arg->prevpiv = dplasma_low_greedy_prevpiv;

    mt = arg->ldd;
    a = arg->a;
    p = arg->p;
    pa = p * a;
    domino = arg->domino;

    if ( domino )
    {
        int j, k, height, start, end, firstk = 0;
        int *nT, *nZ;

        arg->minMN =  min( minMN, mt*a );
        minMN = arg->minMN;

        arg->ipiv = (int*)malloc( mt * minMN * sizeof(int) );
        ipiv = arg->ipiv;
        memset(ipiv, 0, mt*minMN*sizeof(int));

        nT = (int*)malloc(minMN*sizeof(int));
        nZ = (int*)malloc(minMN*sizeof(int));

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
    else
    {
        int j, k, myrank, height, start, end, firstk = 0;
        int *nT2DO = (int*)malloc(minMN*sizeof(int));
        int *nT = (int*)malloc(minMN*sizeof(int));
        int *nZ = (int*)malloc(minMN*sizeof(int));

        arg->ipiv = (int*)malloc( mt * minMN * sizeof(int) * p );
        ipiv = arg->ipiv;

        memset( ipiv,  0, minMN*sizeof(int)*mt*p);

        for ( myrank=0; myrank<p; myrank++ ) {
            int lminMN = minMN;

            memset( nT2DO, 0, minMN*sizeof(int));
            memset( nT,    0, minMN*sizeof(int));
            memset( nZ,    0, minMN*sizeof(int));

            nT[0] = mt;

            for(k=0; k<lminMN; k++) {
                nT2DO[k] = max( mt - ((k + p - 1 - myrank) / pa), 0 );
                if ( nT2DO[k] == 0 ) {
                    lminMN = k;
                    break;
                }
            }

            k = 0; firstk = 0;
            while ( (!( ( nT[lminMN-1] == nT2DO[lminMN-1] ) &&
                        ( nZ[lminMN-1]+1 == nT[lminMN-1] ) ) )
                    && ( firstk < lminMN ) ) {
                height = (nT[k] - nZ[k]) / 2;
                if ( height == 0 ) {
                    while ( (firstk < lminMN) &&
                            ( nT[firstk] == nT2DO[firstk] ) &&
                            ( nZ[firstk]+1 == nT[firstk] ) ) {
                        if (  ( firstk < lminMN-1 )  &&
                              (( (firstk) % pa) != ((a-1)*p+myrank) ) )
                            nT[firstk+1]++;
                        firstk++;
                    }
                    k = firstk;
                    continue;
                }

                if (k < lminMN-1) nT[k+1] += height;
                start = mt - nZ[k] - 1;
                end = start - height;
                nZ[k] += height;

                for( j=start; j > end; j-- ) {
                    ipiv[ myrank*mt*minMN + k*mt + j ] = (j - height);
                }

                k++;
                if (k > lminMN-1) k = firstk;
            }
        }

        free(nT2DO);
        free(nT);
        free(nZ);
    }

#if 0
    {
        int m, k;
        for(m=0; m<mt; m++) {
            printf("%3d | ", m);
            for (k=0; k<minMN; k++) {
                printf( "%3d ", ipiv[k*mt + m] );
            }
            printf("\n");
        }
    }
    if (!arg->domino) {
        int m, k, myrank;
        for ( myrank=1; myrank<p; myrank++ ) {
            ipiv += mt*minMN;
            printf("-------- rank %d ---------\n", myrank );
            for(m=0; m<mt; m++) {
                printf("%3d | ", m);
                for (k=0; k<minMN; k++) {
                  int k_a = (k + p - 1 - myrank) / p / a;
                  if ( m >= k_a )
                    printf( "%3d ", ipiv[k*mt + m] );
                  else
                    printf( "--- " );
                }
                printf("\n");
            }
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
        else if ( start < arg->ldd && (start-k < arg->p-1) )
            return start+1;
    }
    return arg->ldd;
};

static int dplasma_high_flat_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{
    assert( arg->p > 1 );
    if ( p == k && arg->ldd > 1 ) {
        if ( start == p && p != arg->ldd-1 )
            return min( p + arg->p - 1, arg->ldd - 1 );
        else if ( start > p + 1 && (start-k < arg->p))
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
 *                 DPLASMA_HIGH_BINARY_TREE
 ***************************************************/
static int dplasma_high_binary_currpiv(const qr_subpiv_t *arg, const int m, const int k)
{
    int tmp1 = m - k;
    int tmp2 = 1;
    (void)arg;

    if ( tmp1 == 0)
        return 0;
    while( (tmp1 != 0 ) && (tmp1 % 2 == 0) ) {
        tmp1 = tmp1 >> 1;
        tmp2 = tmp2 << 1;
    }
    assert( m - tmp2 >= k );
    return m - tmp2;
};

static int dplasma_high_binary_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{
    int tmpp, bit;
    myassert( (start == arg->ldd) || (dplasma_high_binary_currpiv( arg, start, k ) == p) );

    if ( start <= p )
        return arg->ldd;

    int offset = p - k;
    bit = 0;
    if (start != arg->ldd) {
        while( ( (start - k) & (1 << bit ) ) == 0 )
            bit++;
        bit++;
    }

    tmpp = offset | (1 << bit);
    if ( ( tmpp != offset ) && (tmpp < arg->p) && ( tmpp+k < arg->ldd ) )
        return tmpp + k;
    else
        return arg->ldd;
};

static int dplasma_high_binary_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{
    int offset = p - k;

    myassert( start >= p && ( start == p || dplasma_high_binary_currpiv( arg, start, k ) == p ) );

    if ( (start == p) && ( offset%2 == 0 ) ) {
        int i, bit, tmp;
        if ( offset == 0 )
            bit = (int)( log( (double)( min(arg->p, arg->ldd - k) ) ) / log( 2. ) );
        else {
            bit = 0;
            while( (offset & (1 << bit )) == 0 )
                bit++;
        }
        for( i=bit; i>-1; i--){
            tmp = offset | (1 << i);
            if ( ( offset != tmp ) && ( tmp < arg->p ) && (tmp+k < arg->ldd) )
                return tmp+k;
        }
        return arg->ldd;
    }

    if ( (start - p) > 1 )
        return p + ( (start - p) >> 1 );
    else {
        return arg->ldd;
    }
};

static void dplasma_high_binary_init(qr_subpiv_t *arg){
    arg->currpiv = dplasma_high_binary_currpiv;
    arg->nextpiv = dplasma_high_binary_nextpiv;
    arg->prevpiv = dplasma_high_binary_prevpiv;
    arg->ipiv = NULL;
};

/****************************************************
 *          DPLASMA_HIGH_FIBONACCI_TREE
 ***************************************************/
/* Return the pivot to use for the row m at step k */
inline static int dplasma_high_fibonacci_currpiv( const qr_subpiv_t *qrpiv, const int m, const int k ) {
    return (qrpiv->ipiv)[ m-k ] + k;
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int dplasma_high_fibonacci_prevpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    myassert( p >= k && start >= p && start-k <= qrpiv->p);

    int lp    = p - k;
    int lstart= start - k;
    int end   = min(qrpiv->ldd-k, qrpiv->p);
    for( i=lstart+1; i<end; i++ )
        if ( (qrpiv->ipiv)[i] == lp )
            return i+k;
    return qrpiv->ldd;
}

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int dplasma_high_fibonacci_nextpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    myassert( p>=k && (start == qrpiv->ldd || start-k <= qrpiv->p) );

    for( i=min(start-k-1, qrpiv->p-1); i>0; i-- )
        if ( (qrpiv->ipiv)[i] == (p-k) )
            return i + k;
    return (qrpiv->ldd);
}

static void dplasma_high_fibonacci_init(qr_subpiv_t *arg){
    int *ipiv;
    int p;

    arg->currpiv = dplasma_high_fibonacci_currpiv;
    arg->nextpiv = dplasma_high_fibonacci_nextpiv;
    arg->prevpiv = dplasma_high_fibonacci_prevpiv;

    p = arg->p;

    arg->ipiv = (int*)malloc( p * sizeof(int) );
    ipiv = arg->ipiv;
    memset(ipiv, 0, p*sizeof(int));

    /*
     * Fibonacci of order 1:
     *    u_(n+1) = u_(n) + 1
     */
    {
        int f1, k, m;

        /* Fill in the first column */
        f1 = 1;
        for (m=1; m < p; ) {
            for (k=0; (k < f1) && (m < p); k++, m++) {
                ipiv[m] = m - f1;
            }
            f1++;
        }
    }
};

/****************************************************
 *                 DPLASMA_HIGH_GREEDY_TREE (1 panel duplicated)
 ***************************************************/
static void dplasma_high_greedy1p_init(qr_subpiv_t *arg){
    int *ipiv;
    int mt, p;

    arg->currpiv = dplasma_high_fibonacci_currpiv;
    arg->nextpiv = dplasma_high_fibonacci_nextpiv;
    arg->prevpiv = dplasma_high_fibonacci_prevpiv;

    mt = arg->ldd;
    p = arg->p;

    arg->ipiv = (int*)malloc( p * sizeof(int) );
    ipiv = arg->ipiv;
    memset(ipiv, 0, p*sizeof(int));

    {
      int minMN = 1;
        int j, k, height, start, end, firstk = 0;
        int *nT = (int*)malloc(minMN*sizeof(int));
        int *nZ = (int*)malloc(minMN*sizeof(int));
        memset( nT, 0, minMN*sizeof(int));
        memset( nZ, 0, minMN*sizeof(int));

        nT[0] = mt;
        nZ[0] = max( mt - p, 0 );
        for(k=1; k<minMN; k++) {
            height = max(mt-k-p, 0);
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
                ipiv[ k*p + j-k ] = (j - height);
            }

            k++;
            if (k > minMN-1) k = firstk;
        }

        free(nT);
        free(nZ);
    }
};

/****************************************************
 *                 DPLASMA_HIGH_GREEDY_TREE
 ***************************************************/
static int dplasma_high_greedy_currpiv(const qr_subpiv_t *arg, const int m, const int k)
{
    myassert( m >= k && m < k+arg->p );
    return (arg->ipiv)[ k * (arg->p) + (m - k) ];
};

static int dplasma_high_greedy_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{
    int i;
    myassert( (start >= k && start < k+arg->p) || start == arg->ldd );
    for( i=min(start-1, k+arg->p-1); i > k; i-- )
        if ( (arg->ipiv)[i-k + k* (arg->p)] == p )
            return i;
    return (arg->ldd);
};

static int dplasma_high_greedy_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{
    int i;
    myassert( (start >= k && start < k+arg->p) || start == p );
    for( i=start-k+1; i<arg->p; i++ )
        if ( (arg->ipiv)[i +  k * (arg->p)] == p )
            return k+i;
    return arg->ldd;
};

static void dplasma_high_greedy_init(qr_subpiv_t *arg, int minMN){
    int *ipiv;
    int mt, p;

    arg->currpiv = dplasma_high_greedy_currpiv;
    arg->nextpiv = dplasma_high_greedy_nextpiv;
    arg->prevpiv = dplasma_high_greedy_prevpiv;

    mt = arg->ldd;
    p = arg->p;

    arg->ipiv = (int*)malloc( p * minMN * sizeof(int) );
    ipiv = arg->ipiv;
    memset(ipiv, 0, p*minMN*sizeof(int));

    {
        int j, k, height, start, end, firstk = 0;
        int *nT = (int*)malloc(minMN*sizeof(int));
        int *nZ = (int*)malloc(minMN*sizeof(int));
        memset( nT, 0, minMN*sizeof(int));
        memset( nZ, 0, minMN*sizeof(int));

        nT[0] = mt;
        nZ[0] = max( mt - p, 0 );
        for(k=1; k<minMN; k++) {
            height = max(mt-k-p, 0);
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
                ipiv[ k*p + j-k ] = (j - height);
            }

            k++;
            if (k > minMN-1) k = firstk;
        }

        free(nT);
        free(nZ);
    }
};

/****************************************************
 *
 *   Generic functions currpiv,prevpiv,nextpiv
 *
 ***************************************************/
int dplasma_qr_currpiv(const qr_piv_t *arg, const int m, const int k)
{
    int tmp, tmpk, perm_m;
    int lm, rank, *perm;
    int a    = arg->a;
    int p    = arg->p;
    int domino = arg->domino;

    perm_m = dplasma_qr_getinvperm( arg, k, m );
    lm   = perm_m / p; /* Local index in the distribution over p domains */
    rank = perm_m % p; /* Staring index in this distribution             */
    perm = arg->perm + (arg->desc->mt+1) * k;

    myassert( (p==1) || (perm_m / (p*a)) == (m / (p*a)) );
    myassert( (p==1) || (perm_m % p) == (m % p) );

    /* TS level common to every case */
    if ( domino ) {
        switch( dplasma_qr_gettype( arg, k, m ) )
        {
        case 0:
            tmp = lm / a;
            if ( tmp == k / arg->a )
                return perm[       k * p + rank ]; /* Below to the first bloc including the diagonal */
            else
                return perm[ tmp * a * p + rank ];
            break;
        case 1:
            tmp = arg->llvl->currpiv(arg->llvl, perm_m, k);
            return perm[ ( tmp == k / arg->a ) ? k * p + rank : tmp * a * p + rank ];
            break;
        case 2:
            return m - p;
            break;
        case 3:
            if ( arg->hlvl != NULL )
                return arg->hlvl->currpiv(arg->hlvl, perm_m, k);
        default:
            return arg->desc->mt;
        }
    }
    else {
        switch( dplasma_qr_gettype( arg, k, m ) )
        {
        case 0:
            tmp = lm / a;
            /* tmpk = (k + p - 1 - m%p) / p / a;  */
            tmpk = k / (p * a);
            return perm[ ( tmp == tmpk ) ? k + (perm_m-k)%p : tmp * a * p + rank ];
            break;
        case 1:
            tmp = arg->llvl->currpiv(arg->llvl, perm_m, k);
            /* tmpk = (k + p - 1 - m%p) / p / a; */
            tmpk = k / (p * a);
            return perm[ ( tmp == tmpk ) ? k + (perm_m-k)%p : tmp * a * p + rank ];
            break;
        case 2:
            return perm[ perm_m - p];
            break;
        case 3:
            if ( arg->hlvl != NULL )
                return perm[arg->hlvl->currpiv(arg->hlvl, perm_m, k)];
        default:
            return arg->desc->mt;
        }
    }
};

int dplasma_qr_nextpiv(const qr_piv_t *arg, int pivot, const int k, int start)
{
    int tmp, ls, lp, nextp;
    int opivot, ostart; /* original values before permutation */
    int lpivot, rpivot, lstart, *perm;
    int a = arg->a;
    int p = arg->p;

    /* fprintf(stderr, "Before: k=%d, pivot=%d, start=%d\n", k, pivot, start); */
    ostart = start;
    opivot = pivot;
    start = dplasma_qr_getinvperm( arg, k, ostart);
    pivot = dplasma_qr_getinvperm( arg, k, opivot);

    /* fprintf(stderr, "After: k=%d, pivot=%d, start=%d\n", k, pivot, start); */

    lpivot = pivot / p; /* Local index in the distribution over p domains */
    rpivot = pivot % p; /* Staring index in this distribution             */

    /* Local index in the distribution over p domains */
    lstart = ( start == arg->desc->mt ) ? arg->llvl->ldd * a : start / p;

    perm = arg->perm + (arg->desc->mt+1) * k;

    myassert( start > pivot && pivot >= k );
    myassert( start == arg->desc->mt || opivot == dplasma_qr_currpiv( arg, ostart, k ) );

    /* TS level common to every case */
    ls = (start < arg->desc->mt) ? dplasma_qr_gettype( arg, k, ostart ) : -1;
    lp = dplasma_qr_gettype( arg, k, opivot );

    switch( ls )
        {
        case -1:

            if ( lp == DPLASMA_QR_KILLED_BY_TS ) {
                myassert( start == arg->desc->mt );
                return arg->desc->mt;
            }

        case DPLASMA_QR_KILLED_BY_TS:

            /* If the tile is over the diagonal of step k, skip directly to type 2 */
            if ( arg->domino && lpivot < k )
                goto next_2;

            if ( start == arg->desc->mt )
                nextp = pivot + p;
            else
                nextp = start + p;

            if ( ( nextp < arg->desc->mt ) &&
                 ( nextp < pivot + a*p ) &&
                 ( (nextp/p)%a != 0 ) )
                return perm[nextp];
            start = arg->desc->mt;
            lstart = arg->llvl->ldd * a;

        case DPLASMA_QR_KILLED_BY_LOCALTREE:

            /* If the tile is over the diagonal of step k, skip directly to type 2 */
            if ( arg->domino && lpivot < k )
                goto next_2;

            /* Get the next pivot for the low level tree */
            tmp = arg->llvl->nextpiv(arg->llvl, pivot, k, lstart / a );

            if ( (tmp * a * p + rpivot >= arg->desc->mt)
                 && (tmp == arg->llvl->ldd-1) )
                tmp = arg->llvl->nextpiv(arg->llvl, pivot, k, tmp);

            if ( tmp != arg->llvl->ldd )
                return perm[tmp * a * p + rpivot];

        next_2:
            /* no next of type 1, we reset start to search the next 2 */
            start = arg->desc->mt;
            lstart = arg->llvl->ldd * a;

        case DPLASMA_QR_KILLED_BY_DOMINO:

            if ( lp < DPLASMA_QR_KILLED_BY_DOMINO ) {
                return arg->desc->mt;
            }

            /* Type 2 are killed only once if they are strictly in the band */
            if ( arg->domino &&
                 (start == arg->desc->mt) &&
                 (lpivot < k)             &&
                 (pivot+p < arg->desc->mt) ) {
                return perm[pivot+p];
            }

            /* no next of type 2, we reset start to search the next 3 */
            start = arg->desc->mt;
            lstart = arg->llvl->ldd * a;

        case DPLASMA_QR_KILLED_BY_DISTTREE:

            if ( lp < DPLASMA_QR_KILLED_BY_DISTTREE ) {
                return arg->desc->mt;
            }

            if( arg->hlvl != NULL ) {
                tmp = arg->hlvl->nextpiv( arg->hlvl, pivot, k, start );
                if ( tmp != arg->desc->mt )
                    return perm[tmp];
            }

        default:
            return arg->desc->mt;
        }
}

int dplasma_qr_nexttriangle(const qr_piv_t *arg, int p, const int k, int m)
{
    int next = dplasma_qr_nextpiv(arg, p, k, m);

    while (dplasma_qr_gettype(arg, k, next) == 0) {
        next = dplasma_qr_nextpiv(arg, p, k, next);
    }
    
    return next;
};

int dplasma_qr_prevpiv(const qr_piv_t *arg, int pivot, const int k, int start)
{
    int tmp, ls, lp, nextp;
    int opivot, ostart; /* original values before permutation */
    int lpivot, rpivot, lstart, *perm;
    int a = arg->a;
    int p = arg->p;

    ostart = start;
    opivot = pivot;
    start = dplasma_qr_getinvperm( arg, k, ostart );
    pivot = dplasma_qr_getinvperm( arg, k, opivot );

    lpivot = pivot / p; /* Local index in the distribution over p domains */
    rpivot = pivot % p; /* Staring index in this distribution             */
    lstart = start / p; /* Local index in the distribution over p domains */
    perm = arg->perm + (arg->desc->mt+1) * k;

    myassert( start >= pivot && pivot >= k && start < arg->desc->mt );
    myassert( start == pivot || opivot == dplasma_qr_currpiv( arg, ostart, k ) );

    /* T Slevel common to every case */
    ls = dplasma_qr_gettype( arg, k, ostart );
    lp = dplasma_qr_gettype( arg, k, opivot );

    if ( lp == DPLASMA_QR_KILLED_BY_TS )
      return arg->desc->mt;

    myassert( lp >= ls );
    switch( ls )
        {
        case DPLASMA_QR_KILLED_BY_DISTTREE:
            if( arg->hlvl != NULL ) {
                tmp = arg->hlvl->prevpiv( arg->hlvl, pivot, k, start );
                if ( tmp != arg->desc->mt )
                    return perm[tmp];
            }

            start = pivot;
            lstart = pivot / p;

        case DPLASMA_QR_KILLED_BY_DOMINO:
            /* If the tile is over the diagonal of step k, process it as type 2 */
            if ( arg->domino && lpivot < k ) {

                if ( ( start == pivot ) &&
                     (start+p < arg->desc->mt ) )
                    return perm[start+p];

                if ( lp > DPLASMA_QR_KILLED_BY_LOCALTREE )
                    return arg->desc->mt;
            }

            start = pivot;
            lstart = pivot / p;

            /* If it is the 'local' diagonal block, we go to 1 */

        case DPLASMA_QR_KILLED_BY_LOCALTREE:
            /* If the tile is over the diagonal of step k and is of type 2,
               it cannot annihilate type 0 or 1 */
            if ( arg->domino && lpivot < k )
                return arg->desc->mt;

            tmp = arg->llvl->prevpiv(arg->llvl, pivot, k, lstart / a);

            if ( (tmp * a * p + rpivot >= arg->desc->mt)
                 && (tmp == arg->llvl->ldd-1) )
                tmp = arg->llvl->prevpiv(arg->llvl, pivot, k, tmp);

            if ( tmp != arg->llvl->ldd )
                return perm[tmp * a * p + rpivot];

            start = pivot;

        case DPLASMA_QR_KILLED_BY_TS:
            /* Search for predecessor in TS tree */
            /* if ( ( start+p < arg->desc->mt ) &&  */
            /*      ( (((start+p) / p) % a) != 0 ) ) */
            /*     return perm[start + p]; */

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
                return perm[nextp];

        default:
            return arg->desc->mt;
        }
};

/****************************************************
 *
 * Generate the permutation required for the round-robin on TS
 *
 ***************************************************/
static void dplasma_qr_genperm( qr_piv_t *qrpiv )
{
    int m = qrpiv->desc->mt;
    int n = qrpiv->desc->nt;
    int a = qrpiv->a;
    int p = qrpiv->p;
    int domino = qrpiv->domino;
    int minMN = min( m, n );
    int pa = p * a;
    int i, j, k;
    int nbextra1;
    int end2;
    int mpa   = m % pa;
    int endpa = m - mpa;
    int *perm;

    qrpiv->perm = (int*)malloc( (m+1) * minMN * sizeof(int) );
    perm = qrpiv->perm;

    if ( qrpiv->tsrr ) {
        for(k=0; k<minMN; k++) {
            for( i=0; i<m+1; i++) {
                perm[i] = -1;
            }
            perm += m+1;
        }
        perm = qrpiv->perm;
        for(k=0; k<minMN; k++) {
            nbextra1 = nbextra1_formula;

            end2 = p + ( domino ? k*p : k + nbextra1 );
            end2 = (( end2 + pa - 1 ) / pa ) * pa;
            end2 = min( end2, m );

            /*
             * All tiles of type 3, 2 and:
             * - 1 when domino is disabled
             * - 0 before the first multiple of pa under the considered diagonal
             */
            for( i=k; i<end2; i++) {
                perm[i] = i;
            }

            /* All permutations in groups of pa tiles */
            assert( i%pa == 0 || i>=endpa);
            for( ; i<endpa; i+=pa ) {
                for(j=0; j<pa; j++) {
                    perm[i+j] = i + ( j + p * (k%a) )%pa;
                }
            }

            /* Last group of tiles */
            if ( i < m ) {
                int lp, la;
                for(lp=0; lp<p; lp++) {
                    la = mpa / p;
                    if ( lp < mpa%p ) la++;

                    for( j=lp; j<mpa && (i+j)<m; j+=p ) {
                        perm[i+j] = i + ( j + p * (k%la) )%(p*la);
                        assert(perm[i+j] < m);
                    }
                }
            }
            perm[m] = m;
            perm += m+1;
        }
    }
    else {
        for(k=0; k<minMN; k++) {
            for( i=0; i<m+1; i++) {
                perm[i] = i;
            }
            perm += m+1;
        }
    }
}

int dplasma_qr_getinvperm( const qr_piv_t *qrpiv, int k, int m )
{
    int p  = qrpiv->p;
    int pa = qrpiv->a * qrpiv->p;
    int start = m / pa * pa;
    int stop  = min( start + pa, qrpiv->desc->mt+1 ) - start;
    int *perm = qrpiv->perm + (qrpiv->desc->mt+1)*k + start;
    int i;

    if (qrpiv->a == 1)
        return m;

    for ( i=m%p; i < stop; i+=p ) {
        if( perm[i] == m )
            return i+start;
    }

    /* We should never arrive here */
    myassert( 0 );
}

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

    a = max( a, 1 );
    p = max( p, 1 );
    domino = domino ? 1 : 0;

    qrpiv->desc = A;
    qrpiv->a = a;
    qrpiv->p = p;
    qrpiv->domino = domino;
    qrpiv->tsrr = tsrr;
    qrpiv->perm = NULL;

    qrpiv->llvl = (qr_subpiv_t*) malloc( sizeof(qr_subpiv_t) );
    qrpiv->hlvl = NULL;

    minMN = min(A->mt, A->nt);
    low_mt = (A->mt + p * a - 1) / ( p * a );

    qrpiv->llvl->minMN  = minMN;
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

        qrpiv->llvl->minMN  = minMN;
        qrpiv->hlvl->ldd    = A->mt;
        qrpiv->hlvl->a      = a;
        qrpiv->hlvl->p      = p;
        qrpiv->hlvl->domino = domino;

        switch( type_hlvl ) {
        case DPLASMA_GREEDY1P_TREE :
            dplasma_high_greedy1p_init(qrpiv->hlvl);
            break;
        case DPLASMA_GREEDY_TREE :
            dplasma_high_greedy_init(qrpiv->hlvl, minMN);
            break;
        case DPLASMA_FIBONACCI_TREE :
            dplasma_high_fibonacci_init(qrpiv->hlvl);
            break;
        case DPLASMA_BINARY_TREE :
            dplasma_high_binary_init(qrpiv->hlvl);
            break;
        case DPLASMA_FLAT_TREE :
        default:
            dplasma_high_flat_init(qrpiv->hlvl);
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
