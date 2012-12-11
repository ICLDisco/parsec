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

static inline int dague_imin(int a, int b) { return (a <= b) ? a : b; };
static inline int dague_imax(int a, int b) { return (a >= b) ? a : b; };

#define PRINT_PIVGEN 0
#ifdef PRINT_PIVGEN
#define myassert( test ) {if ( ! (test) ) return -1;}
#else
#define myassert(test) {assert((test)); return -1;}
#endif


struct hqr_args_s;
typedef struct hqr_args_s hqr_args_t;

struct hqr_subpiv_s;
typedef struct hqr_subpiv_s hqr_subpiv_t;

struct hqr_args_s {
    int domino;  /* Switch to enable/disable the domino tree linking high and lw level reduction trees */
    int tsrr;    /* Switch to enable/disable round-robin on TS to optimise pipelining between TS and local tree */
    hqr_subpiv_t *llvl;
    hqr_subpiv_t *hlvl;
    int *perm;
};

struct hqr_subpiv_s {
    /*
     * currpiv
     *    @param[in] arg pointer to the qr_piv structure
     *    @param[in] m   line you want to eliminate
     *    @param[in] k   step in the factorization
     *
     *  @return the annihilator p used with m at step k
     */
    int (*currpiv)(const hqr_subpiv_t *arg, int k, int m);
    /*
     * nextpiv
     *    @param[in] arg pointer to the qr_piv structure
     *    @param[in] p   line currently used as an annihilator
     *    @param[in] k   step in the factorization
     *    @param[in] m   line actually annihilated.
     *          m = MT to find the first time p is used as an annihilator during step k
     *
     *  @return the next line m' using the line p as annihilator during step k
     *          desc->mt if p will never be used again as an annihilator.
     */
    int (*nextpiv)(const hqr_subpiv_t *arg, int k, int p, int m);
    /*
     * nextpiv
     *    @param[in] arg pointer to the qr_piv structure
     *    @param[in] p   line currently used as an annihilator
     *    @param[in] k   step in the factorization
     *    @param[in] m   line actually annihilated.
     *          m = p to find the last time p has been used as an annihilator during step k
     *
     *  @return the previous line m' using the line p as annihilator during step k
     *          desc->mt if p has never been used before that as an annihilator.
     */
    int (*prevpiv)(const hqr_subpiv_t *arg, int k, int p, int m);
    int *ipiv;
    int minMN;
    int ldd;
    int a;
    int p;
    int domino;
};


/*
 * Common functions
 */
static int hqr_getnbgeqrf( const dplasma_qrtree_t *qrtree, int k );
static int hqr_getm(       const dplasma_qrtree_t *qrtree, int k, int i   );
static int hqr_geti(       const dplasma_qrtree_t *qrtree, int k, int m   );
static int hqr_gettype(    const dplasma_qrtree_t *qrtree, int k, int m   );

/* Permutation */
static void hqr_genperm   (       dplasma_qrtree_t *qrtree );
static int  hqr_getinvperm( const dplasma_qrtree_t *qrtree, int k, int m );

/*
 * Subtree for low-level
 */
static void hqr_low_flat_init(     hqr_subpiv_t *arg);
static void hqr_low_greedy_init(   hqr_subpiv_t *arg, int minMN);
static void hqr_low_binary_init(   hqr_subpiv_t *arg);
static void hqr_low_fibonacci_init(hqr_subpiv_t *arg, int minMN);

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
static int hqr_getnbgeqrf( const dplasma_qrtree_t *qrtree, int k ) {
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int a = qrtree->a;
    int p = qrtree->p;
    int gmt = qrtree->desc->mt;
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
    nb_1 += dague_imin( p, gmt - nb_12 );

    return dague_imin( nb_1 + nb_2 + nb_3, gmt - k);
}

/*
 * Extra parameter:
 *    i - indice of the geqrt in the continuous space
 * Return:
 *    The global indice m of the i th geqrt in the panel k
 */
static int hqr_getm( const dplasma_qrtree_t *qrtree, int k, int i )
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int a = qrtree->a;
    int p = qrtree->p;
    int gmt = qrtree->desc->mt + 1;
    int domino = arg->domino;
    int *perm  = arg->perm + gmt * k;

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
static int hqr_geti( const dplasma_qrtree_t *qrtree, int k, int m )
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int a = qrtree->a;
    int p = qrtree->p;
    int domino = arg->domino;
    int lm = hqr_getinvperm( qrtree, k, m );

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
static int hqr_gettype( const dplasma_qrtree_t *qrtree, int k, int m ) {
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int a = qrtree->a;
    int p = qrtree->p;
    int domino = arg->domino;

    int lm = hqr_getinvperm( qrtree, k, m );
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

/****************************************************
 *                 HQR_LOW_FLAT_TREE
 ***************************************************/
static int hqr_low_flat_currpiv(const hqr_subpiv_t *arg, int k, int m)
{
    (void)m;
    if ( arg->domino )
        return k / arg->a;
    else
        return (k + arg->p - 1 - m%(arg->p)) / arg->p / arg->a ;
};

static int hqr_low_flat_nextpiv(const hqr_subpiv_t *arg, int k, int p, int start_pa)
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

static int hqr_low_flat_prevpiv(const hqr_subpiv_t *arg, int k, int p, int start_pa)
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

static void hqr_low_flat_init(hqr_subpiv_t *arg){
    arg->currpiv = hqr_low_flat_currpiv;
    arg->nextpiv = hqr_low_flat_nextpiv;
    arg->prevpiv = hqr_low_flat_prevpiv;
    arg->ipiv = NULL;
};

/****************************************************
 *                 HQR_LOW_BINARY_TREE
 ***************************************************/
static int hqr_low_binary_currpiv(const hqr_subpiv_t *arg, int k, int m)
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

static int hqr_low_binary_nextpiv(const hqr_subpiv_t *arg, int k, int p, int start_pa)
{
    int k_a = arg->domino ? k / arg->a :  (k + arg->p - 1 - p%(arg->p)) / arg->p / arg->a;
    int p_pa = (p / arg->p ) / arg->a;

    int tmpp, bit;
    myassert( (start_pa == arg->ldd) || (hqr_low_binary_currpiv( arg, k, start_pa*arg->a*arg->p ) == p_pa || !arg->domino) );

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

static int hqr_low_binary_prevpiv(const hqr_subpiv_t *arg, int k, int p, int start_pa)
{
    int k_a = arg->domino ? k / arg->a :  (k + arg->p - 1 - p%(arg->p)) / arg->p / arg->a;
    int p_pa = (p / arg->p ) / arg->a;
    int offset = p_pa - k_a;

    myassert( start_pa >= p_pa && ( start_pa == p_pa || !arg->domino ||
                                    hqr_low_binary_currpiv( arg, k, start_pa*arg->a*arg->p ) == p_pa ) );

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

static void hqr_low_binary_init(hqr_subpiv_t *arg){
    arg->currpiv = hqr_low_binary_currpiv;
    arg->nextpiv = hqr_low_binary_nextpiv;
    arg->prevpiv = hqr_low_binary_prevpiv;
    arg->ipiv = NULL;
};

/****************************************************
 *          HQR_LOW_FIBONACCI_TREE
 ***************************************************/
/* Return the pivot to use for the row m at step k */
inline static int hqr_low_fibonacci_currpiv( const hqr_subpiv_t *qrpiv, int k, int m ) {
    int k_a = qrpiv->domino ? k / qrpiv->a : (k + qrpiv->p - 1 - m%(qrpiv->p)) / qrpiv->p / qrpiv->a;
    return (qrpiv->ipiv)[ k_a * (qrpiv->ldd) + ( (m / qrpiv->p) / qrpiv->a ) ];
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int hqr_low_fibonacci_prevpiv( const hqr_subpiv_t *qrpiv, int k, int p, int start_pa ) {
    int i;
    int k_a = qrpiv->domino ? k / qrpiv->a :  (k + qrpiv->p - 1 - p%(qrpiv->p)) / qrpiv->p / qrpiv->a;
    int p_pa = (p / qrpiv->p ) / qrpiv->a;

    for( i=start_pa+1; i<(qrpiv->ldd); i++ )
        if ( (qrpiv->ipiv)[i +  k_a * (qrpiv->ldd)] == p_pa )
            return i;
    return i;
 }

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int hqr_low_fibonacci_nextpiv( const hqr_subpiv_t *qrpiv, int k, int p, int start_pa ) {
    int i;
    int k_a = qrpiv->domino ? k / qrpiv->a :  (k + qrpiv->p - 1 - p%(qrpiv->p)) / qrpiv->p / qrpiv->a;
    int p_pa = (p / qrpiv->p ) / qrpiv->a;

    for( i=start_pa-1; i>k_a; i-- )
        if ( (qrpiv->ipiv)[i + k_a * (qrpiv->ldd)] == p_pa )
            return i;
    return (qrpiv->ldd);
}

static void hqr_low_fibonacci_init(hqr_subpiv_t *arg, int minMN){
    int *ipiv;
    int mt;

    arg->currpiv = hqr_low_fibonacci_currpiv;
    arg->nextpiv = hqr_low_fibonacci_nextpiv;
    arg->prevpiv = hqr_low_fibonacci_prevpiv;

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
 *          HQR_LOW_GREEDY_TREE
 ***************************************************/
/* Return the pivot to use for the row m at step k */
inline static int hqr_low_greedy_currpiv( const hqr_subpiv_t *qrpiv, int k, int m ) {
    if (qrpiv->domino)
        return (qrpiv->ipiv)[ k * (qrpiv->ldd) + ( (m / qrpiv->p) / qrpiv->a ) ];
    else
        return (qrpiv->ipiv)[ ( (m%qrpiv->p) * qrpiv->minMN + k ) * (qrpiv->ldd)
                              + ( ( m  / qrpiv->p ) / qrpiv->a ) ];
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int hqr_low_greedy_prevpiv( const hqr_subpiv_t *qrpiv, int k, int p, int start_pa ) {
    int i;
    int p_pa = p / qrpiv->p / qrpiv->a;
    int *ipiv = qrpiv->domino ? qrpiv->ipiv : qrpiv->ipiv + p%qrpiv->p * qrpiv->minMN *qrpiv->ldd;

    for( i=start_pa+1; i<(qrpiv->ldd); i++ )
        if ( ipiv[i +  k * (qrpiv->ldd)] == p_pa )
            return i;
    return i;
 }

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int hqr_low_greedy_nextpiv( const hqr_subpiv_t *qrpiv, int k, int p, int start_pa ) {
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

static void hqr_low_greedy_init(hqr_subpiv_t *arg, int minMN){
    int *ipiv;
    int mt, a, p, pa, domino;

    arg->currpiv = hqr_low_greedy_currpiv;
    arg->nextpiv = hqr_low_greedy_nextpiv;
    arg->prevpiv = hqr_low_greedy_prevpiv;

    mt = arg->ldd;
    a = arg->a;
    p = arg->p;
    pa = p * a;
    domino = arg->domino;

    if ( domino )
    {
        int j, k, height, start, end, firstk = 0;
        int *nT, *nZ;

        arg->minMN =  dague_imin( minMN, mt*a );
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
                nT2DO[k] = dague_imax( mt - ((k + p - 1 - myrank) / pa), 0 );
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
 *                 HQR_HIGH_FLAT_TREE
 ***************************************************/
static int hqr_high_flat_currpiv(const hqr_subpiv_t *arg, int k, int m)
{
    (void)arg;
    (void)m;
    return k;
};

static int hqr_high_flat_nextpiv(const hqr_subpiv_t *arg, int k, int p, int start)
{
    if ( p == k && arg->ldd > 1 ) {
        if ( start == arg->ldd )
            return p+1;
        else if ( start < arg->ldd && (start-k < arg->p-1) )
            return start+1;
    }
    return arg->ldd;
};

static int hqr_high_flat_prevpiv(const hqr_subpiv_t *arg, int k, int p, int start)
{
    assert( arg->p > 1 );
    if ( p == k && arg->ldd > 1 ) {
        if ( start == p && p != arg->ldd-1 )
            return dague_imin( p + arg->p - 1, arg->ldd - 1 );
        else if ( start > p + 1 && (start-k < arg->p))
            return start-1;
    }
    return arg->ldd;
};

static void hqr_high_flat_init(hqr_subpiv_t *arg){
    arg->currpiv = hqr_high_flat_currpiv;
    arg->nextpiv = hqr_high_flat_nextpiv;
    arg->prevpiv = hqr_high_flat_prevpiv;
    arg->ipiv = NULL;
};

/****************************************************
 *                 HQR_HIGH_BINARY_TREE
 ***************************************************/
static int hqr_high_binary_currpiv(const hqr_subpiv_t *arg, int k, int m)
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

static int hqr_high_binary_nextpiv(const hqr_subpiv_t *arg, int k, int p, int start)
{
    int tmpp, bit;
    myassert( (start == arg->ldd) || (hqr_high_binary_currpiv( arg, k, start ) == p) );

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


static int hqr_high_binary_prevpiv(const hqr_subpiv_t *arg, int k, int p, int start)
{
    int offset = p - k;

    myassert( start >= p && ( start == p || hqr_high_binary_currpiv( arg, k, start ) == p ) );

    if ( (start == p) && ( offset%2 == 0 ) ) {
        int i, bit, tmp;
        if ( offset == 0 )
            bit = (int)( log( (double)( dague_imin(arg->p, arg->ldd - k) ) ) / log( 2. ) );
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

static void hqr_high_binary_init(hqr_subpiv_t *arg){
    arg->currpiv = hqr_high_binary_currpiv;
    arg->nextpiv = hqr_high_binary_nextpiv;
    arg->prevpiv = hqr_high_binary_prevpiv;
    arg->ipiv = NULL;
};

/****************************************************
 *          HQR_HIGH_FIBONACCI_TREE
 ***************************************************/
/* Return the pivot to use for the row m at step k */
inline static int hqr_high_fibonacci_currpiv( const hqr_subpiv_t *qrpiv, int k, int m ) {
    return (qrpiv->ipiv)[ m-k ] + k;
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int hqr_high_fibonacci_prevpiv( const hqr_subpiv_t *qrpiv, int k, int p, int start ) {
    int i;
    myassert( p >= k && start >= p && start-k <= qrpiv->p);

    int lp    = p - k;
    int lstart= start - k;
    int end   = dague_imin(qrpiv->ldd-k, qrpiv->p);
    for( i=lstart+1; i<end; i++ )
        if ( (qrpiv->ipiv)[i] == lp )
            return i+k;
    return qrpiv->ldd;
}

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int hqr_high_fibonacci_nextpiv( const hqr_subpiv_t *qrpiv, int k, int p, int start ) {
    int i;
    myassert( p>=k && (start == qrpiv->ldd || start-k <= qrpiv->p) );

    for( i=dague_imin(start-k-1, qrpiv->p-1); i>0; i-- )
        if ( (qrpiv->ipiv)[i] == (p-k) )
            return i + k;
    return (qrpiv->ldd);
}

static void hqr_high_fibonacci_init(hqr_subpiv_t *arg){
    int *ipiv;
    int p;

    arg->currpiv = hqr_high_fibonacci_currpiv;
    arg->nextpiv = hqr_high_fibonacci_nextpiv;
    arg->prevpiv = hqr_high_fibonacci_prevpiv;

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
 *                 HQR_HIGH_GREEDY_TREE (1 panel duplicated)
 ***************************************************/
static void hqr_high_greedy1p_init(hqr_subpiv_t *arg){
    int *ipiv;
    int mt, p;

    arg->currpiv = hqr_high_fibonacci_currpiv;
    arg->nextpiv = hqr_high_fibonacci_nextpiv;
    arg->prevpiv = hqr_high_fibonacci_prevpiv;

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
        nZ[0] = dague_imax( mt - p, 0 );
        for(k=1; k<minMN; k++) {
            height = dague_imax(mt-k-p, 0);
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
 *                 HQR_HIGH_GREEDY_TREE
 ***************************************************/
static int hqr_high_greedy_currpiv(const hqr_subpiv_t *arg, int k, int m)
{
    myassert( m >= k && m < k+arg->p );
    return (arg->ipiv)[ k * (arg->p) + (m - k) ];
};

static int hqr_high_greedy_nextpiv(const hqr_subpiv_t *arg, int k, int p, int start)
{
    int i;
    myassert( (start >= k && start < k+arg->p) || start == arg->ldd );
    for( i=dague_imin(start-1, k+arg->p-1); i > k; i-- )
        if ( (arg->ipiv)[i-k + k* (arg->p)] == p )
            return i;
    return (arg->ldd);
};

static int hqr_high_greedy_prevpiv(const hqr_subpiv_t *arg, int k, int p, int start)
{
    int i;
    myassert( (start >= k && start < k+arg->p) || start == p );
    for( i=start-k+1; i<arg->p; i++ )
        if ( (arg->ipiv)[i +  k * (arg->p)] == p )
            return k+i;
    return arg->ldd;
};

static void hqr_high_greedy_init(hqr_subpiv_t *arg, int minMN){
    int *ipiv;
    int mt, p;

    arg->currpiv = hqr_high_greedy_currpiv;
    arg->nextpiv = hqr_high_greedy_nextpiv;
    arg->prevpiv = hqr_high_greedy_prevpiv;

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
        nZ[0] = dague_imax( mt - p, 0 );
        for(k=1; k<minMN; k++) {
            height = dague_imax(mt-k-p, 0);
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
int hqr_currpiv(const dplasma_qrtree_t *qrtree, int k, int m)
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int tmp, tmpk, perm_m;
    int lm, rank, *perm;
    int a = qrtree->a;
    int p = qrtree->p;
    int domino = arg->domino;
    int gmt = qrtree->desc->mt;

    perm_m = hqr_getinvperm( qrtree, k, m );
    lm   = perm_m / p; /* Local index in the distribution over p domains */
    rank = perm_m % p; /* Staring index in this distribution             */
    perm = arg->perm + (gmt+1) * k;

    myassert( (p==1) || (perm_m / (p*a)) == (m / (p*a)) );
    myassert( (p==1) || (perm_m % p) == (m % p) );

    /* TS level common to every case */
    if ( domino ) {
        switch( hqr_gettype( qrtree, k, m ) )
        {
        case 0:
            tmp = lm / a;
            if ( tmp == k / a )
                return perm[       k * p + rank ]; /* Below to the first bloc including the diagonal */
            else
                return perm[ tmp * a * p + rank ];
            break;
        case 1:
            tmp = arg->llvl->currpiv(arg->llvl, k, perm_m);
            return perm[ ( tmp == k / a ) ? k * p + rank : tmp * a * p + rank ];
            break;
        case 2:
            return m - p;
            break;
        case 3:
            if ( arg->hlvl != NULL )
                return arg->hlvl->currpiv(arg->hlvl, k, perm_m);
        default:
            return gmt;
        }
    }
    else {
        switch( hqr_gettype( qrtree, k, m ) )
        {
        case 0:
            tmp = lm / a;
            /* tmpk = (k + p - 1 - m%p) / p / a;  */
            tmpk = k / (p * a);
            return perm[ ( tmp == tmpk ) ? k + (perm_m-k)%p : tmp * a * p + rank ];
            break;
        case 1:
            tmp = arg->llvl->currpiv(arg->llvl, k, perm_m);
            /* tmpk = (k + p - 1 - m%p) / p / a; */
            tmpk = k / (p * a);
            return perm[ ( tmp == tmpk ) ? k + (perm_m-k)%p : tmp * a * p + rank ];
            break;
        case 2:
            return perm[ perm_m - p];
            break;
        case 3:
            if ( arg->hlvl != NULL )
                return perm[arg->hlvl->currpiv(arg->hlvl, k, perm_m)];
        default:
            return gmt;
        }
    }
};

/**
 *  hqr_nextpiv - Computes the next row killed by the row p, after
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
static int hqr_nextpiv(const dplasma_qrtree_t *qrtree, int k, int pivot, int start)
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int tmp, ls, lp, nextp;
    int opivot, ostart; /* original values before permutation */
    int lpivot, rpivot, lstart, *perm;
    int a = qrtree->a;
    int p = qrtree->p;
    int gmt = qrtree->desc->mt;

    /* fprintf(stderr, "Before: k=%d, pivot=%d, start=%d\n", k, pivot, start); */
    ostart = start;
    opivot = pivot;
    start = hqr_getinvperm( qrtree, k, ostart);
    pivot = hqr_getinvperm( qrtree, k, opivot);

    /* fprintf(stderr, "After: k=%d, pivot=%d, start=%d\n", k, pivot, start); */

    lpivot = pivot / p; /* Local index in the distribution over p domains */
    rpivot = pivot % p; /* Staring index in this distribution             */

    /* Local index in the distribution over p domains */
    lstart = ( start == gmt ) ? arg->llvl->ldd * a : start / p;

    perm = arg->perm + (gmt+1) * k;

    myassert( start > pivot && pivot >= k );
    myassert( start == gmt || opivot == hqr_currpiv( qrtree, k, ostart ) );

    /* TS level common to every case */
    ls = (start < gmt) ? hqr_gettype( qrtree, k, ostart ) : -1;
    lp = hqr_gettype( qrtree, k, opivot );

    switch( ls )
        {
        case -1:

            if ( lp == DPLASMA_QR_KILLED_BY_TS ) {
                myassert( start == gmt );
                return gmt;
            }

        case DPLASMA_QR_KILLED_BY_TS:

            /* If the tile is over the diagonal of step k, skip directly to type 2 */
            if ( arg->domino && lpivot < k )
                goto next_2;

            if ( start == gmt )
                nextp = pivot + p;
            else
                nextp = start + p;

            if ( ( nextp < gmt ) &&
                 ( nextp < pivot + a*p ) &&
                 ( (nextp/p)%a != 0 ) )
                return perm[nextp];
            start = gmt;
            lstart = arg->llvl->ldd * a;

        case DPLASMA_QR_KILLED_BY_LOCALTREE:

            /* If the tile is over the diagonal of step k, skip directly to type 2 */
            if ( arg->domino && lpivot < k )
                goto next_2;

            /* Get the next pivot for the low level tree */
            tmp = arg->llvl->nextpiv(arg->llvl, k, pivot, lstart / a );

            if ( (tmp * a * p + rpivot >= gmt)
                 && (tmp == arg->llvl->ldd-1) )
                tmp = arg->llvl->nextpiv(arg->llvl, k, pivot, tmp);

            if ( tmp != arg->llvl->ldd )
                return perm[tmp * a * p + rpivot];

        next_2:
            /* no next of type 1, we reset start to search the next 2 */
            start = gmt;
            lstart = arg->llvl->ldd * a;

        case DPLASMA_QR_KILLED_BY_DOMINO:

            if ( lp < DPLASMA_QR_KILLED_BY_DOMINO ) {
                return gmt;
            }

            /* Type 2 are killed only once if they are strictly in the band */
            if ( arg->domino &&
                 (start == gmt) &&
                 (lpivot < k)             &&
                 (pivot+p < gmt) ) {
                return perm[pivot+p];
            }

            /* no next of type 2, we reset start to search the next 3 */
            start = gmt;
            lstart = arg->llvl->ldd * a;

        case DPLASMA_QR_KILLED_BY_DISTTREE:

            if ( lp < DPLASMA_QR_KILLED_BY_DISTTREE ) {
                return gmt;
            }

            if( arg->hlvl != NULL ) {
                tmp = arg->hlvl->nextpiv( arg->hlvl, k, pivot, start );
                if ( tmp != gmt )
                    return perm[tmp];
            }

        default:
            return gmt;
        }
}

/**
 *  hqr_prevpiv - Computes the previous row killed by the row p, before
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
static int hqr_prevpiv(const dplasma_qrtree_t *qrtree, int k, int pivot, int start)
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int tmp, ls, lp, nextp;
    int opivot, ostart; /* original values before permutation */
    int lpivot, rpivot, lstart, *perm;
    int a = qrtree->a;
    int p = qrtree->p;
    int gmt = qrtree->desc->mt;

    ostart = start;
    opivot = pivot;
    start = hqr_getinvperm( qrtree, k, ostart );
    pivot = hqr_getinvperm( qrtree, k, opivot );

    lpivot = pivot / p; /* Local index in the distribution over p domains */
    rpivot = pivot % p; /* Staring index in this distribution             */
    lstart = start / p; /* Local index in the distribution over p domains */
    perm = arg->perm + (gmt+1) * k;

    myassert( start >= pivot && pivot >= k && start < gmt );
    myassert( start == pivot || opivot == hqr_currpiv( qrtree, k, ostart ) );

    /* T Slevel common to every case */
    ls = hqr_gettype( qrtree, k, ostart );
    lp = hqr_gettype( qrtree, k, opivot );

    if ( lp == DPLASMA_QR_KILLED_BY_TS )
      return gmt;

    myassert( lp >= ls );
    switch( ls )
        {
        case DPLASMA_QR_KILLED_BY_DISTTREE:
            if( arg->hlvl != NULL ) {
                tmp = arg->hlvl->prevpiv( arg->hlvl, k, pivot, start );
                if ( tmp != gmt )
                    return perm[tmp];
            }

            start = pivot;
            lstart = pivot / p;

        case DPLASMA_QR_KILLED_BY_DOMINO:
            /* If the tile is over the diagonal of step k, process it as type 2 */
            if ( arg->domino && lpivot < k ) {

                if ( ( start == pivot ) &&
                     (start+p < gmt ) )
                    return perm[start+p];

                if ( lp > DPLASMA_QR_KILLED_BY_LOCALTREE )
                    return gmt;
            }

            start = pivot;
            lstart = pivot / p;

            /* If it is the 'local' diagonal block, we go to 1 */

        case DPLASMA_QR_KILLED_BY_LOCALTREE:
            /* If the tile is over the diagonal of step k and is of type 2,
               it cannot annihilate type 0 or 1 */
            if ( arg->domino && lpivot < k )
                return gmt;

            tmp = arg->llvl->prevpiv(arg->llvl, k, pivot, lstart / a);

            if ( (tmp * a * p + rpivot >= gmt)
                 && (tmp == arg->llvl->ldd-1) )
                tmp = arg->llvl->prevpiv(arg->llvl, k, pivot, tmp);

            if ( tmp != arg->llvl->ldd )
                return perm[tmp * a * p + rpivot];

            start = pivot;

        case DPLASMA_QR_KILLED_BY_TS:
            /* Search for predecessor in TS tree */
            /* if ( ( start+p < gmt ) &&  */
            /*      ( (((start+p) / p) % a) != 0 ) ) */
            /*     return perm[start + p]; */

            if ( start == pivot ) {
                tmp = lpivot + a - 1 - lpivot%a;
                nextp = tmp * p + rpivot;

                while( pivot < nextp && nextp >= gmt )
                    nextp -= p;
            } else {
                nextp = start - p; /*(lstart - 1) * p + rpivot;*/
            }
            assert(nextp < gmt);
            if ( pivot < nextp )
                return perm[nextp];

        default:
            return gmt;
        }
};

/****************************************************
 *
 * Generate the permutation required for the round-robin on TS
 *
 ***************************************************/
static void hqr_genperm( dplasma_qrtree_t *qrtree )
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int m = qrtree->desc->mt;
    int n = qrtree->desc->nt;
    int a = qrtree->a;
    int p = qrtree->p;
    int domino = arg->domino;
    int minMN = dague_imin( m, n );
    int pa = p * a;
    int i, j, k;
    int nbextra1;
    int end2;
    int mpa   = m % pa;
    int endpa = m - mpa;
    int *perm;

    arg->perm = (int*)malloc( (m+1) * minMN * sizeof(int) );
    perm = arg->perm;

    if ( arg->tsrr ) {
        for(k=0; k<minMN; k++) {
            for( i=0; i<m+1; i++) {
                perm[i] = -1;
            }
            perm += m+1;
        }
        perm = arg->perm;
        for(k=0; k<minMN; k++) {
            nbextra1 = nbextra1_formula;

            end2 = p + ( domino ? k*p : k + nbextra1 );
            end2 = (( end2 + pa - 1 ) / pa ) * pa;
            end2 = dague_imin( end2, m );

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

static int hqr_getinvperm( const dplasma_qrtree_t *qrtree, int k, int m )
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int gmt = qrtree->desc->mt + 1;
    int a = qrtree->a;
    int p = qrtree->p;
    int pa = p * a;
    int start = m / pa * pa;
    int stop  = dague_imin( start + pa, gmt ) - start;
    int *perm = arg->perm + gmt * k + start;
    int i;

    if (a == 1)
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
void dplasma_hqr_init( dplasma_qrtree_t *qrtree,
                       tiled_matrix_desc_t *A,
                       int type_llvl, int type_hlvl,
                       int a, int p,
                       int domino, int tsrr )
{
    int low_mt, minMN;
    hqr_args_t *arg;

    a = dague_imax( a, 1 );
    p = dague_imax( p, 1 );
    domino = domino ? 1 : 0;


    qrtree->getnbgeqrf = hqr_getnbgeqrf;
    qrtree->getm       = hqr_getm;
    qrtree->geti       = hqr_geti;
    qrtree->gettype    = hqr_gettype;
    qrtree->currpiv    = hqr_currpiv;
    qrtree->nextpiv    = hqr_nextpiv;
    qrtree->prevpiv    = hqr_prevpiv;

    qrtree->desc = A;
    qrtree->a    = a;
    qrtree->p    = p;
    qrtree->args = NULL;

    arg = (hqr_args_t*) malloc( sizeof(hqr_args_t) );
    arg->domino = domino;
    arg->tsrr = tsrr;
    arg->perm = NULL;

    arg->llvl = (hqr_subpiv_t*) malloc( sizeof(hqr_subpiv_t) );
    arg->hlvl = NULL;

    minMN = dague_imin(A->mt, A->nt);
    low_mt = (A->mt + p * a - 1) / ( p * a );

    arg->llvl->minMN  = minMN;
    arg->llvl->ldd    = low_mt;
    arg->llvl->a      = a;
    arg->llvl->p      = p;
    arg->llvl->domino = domino;

    switch( type_llvl ) {
    case DPLASMA_GREEDY_TREE :
        hqr_low_greedy_init(arg->llvl, minMN);
        break;
    case DPLASMA_FIBONACCI_TREE :
        hqr_low_fibonacci_init(arg->llvl, minMN);
        break;
    case DPLASMA_BINARY_TREE :
        hqr_low_binary_init(arg->llvl);
        break;
    case DPLASMA_FLAT_TREE :
    default:
        hqr_low_flat_init(arg->llvl);
    }

    if ( p > 1 ) {
        arg->hlvl = (hqr_subpiv_t*) malloc( sizeof(hqr_subpiv_t) );

        arg->llvl->minMN  = minMN;
        arg->hlvl->ldd    = A->mt;
        arg->hlvl->a      = a;
        arg->hlvl->p      = p;
        arg->hlvl->domino = domino;

        switch( type_hlvl ) {
        case DPLASMA_GREEDY1P_TREE :
            hqr_high_greedy1p_init(arg->hlvl);
            break;
        case DPLASMA_GREEDY_TREE :
            hqr_high_greedy_init(arg->hlvl, minMN);
            break;
        case DPLASMA_FIBONACCI_TREE :
            hqr_high_fibonacci_init(arg->hlvl);
            break;
        case DPLASMA_BINARY_TREE :
            hqr_high_binary_init(arg->hlvl);
            break;
        case DPLASMA_FLAT_TREE :
        default:
            hqr_high_flat_init(arg->hlvl);
        }
    }

    qrtree->args = (void*)arg;
    hqr_genperm( qrtree );

    return;
}

void dplasma_hqr_finalize( dplasma_qrtree_t *qrtree )
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);

    if (arg != NULL) {
        if ( arg->llvl != NULL) {
            if ( arg->llvl->ipiv != NULL )
                free( arg->llvl->ipiv );
            free( arg->llvl );
        }

        if ( arg->hlvl != NULL) {
            if ( arg->hlvl->ipiv != NULL )
                free( arg->hlvl->ipiv );
            free( arg->hlvl );
        }

        if ( arg->perm != NULL )
            free(arg->perm);

        free(arg);
    }
}
