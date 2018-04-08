/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
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
#include "dplasma.h"
#include "dplasma_qr_param.h"

#include <math.h>
#if defined(PARSEC_HAVE_STRING_H)
#include <string.h>
#endif  /* defined(PARSEC_HAVE_STRING_H) */

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
     *    @param[in] k   step in the factorization
     *    @param[in] m   line you want to eliminate
     *
     *  @return the annihilator p used with m at step k
     */
    int (*currpiv)(const hqr_subpiv_t *arg, int k, int m);
    /*
     * nextpiv
     *    @param[in] arg pointer to the qr_piv structure
     *    @param[in] k   step in the factorization
     *    @param[in] p   line currently used as an annihilator
     *    @param[in] m   line actually annihilated.
     *          m = MT to find the first time p is used as an annihilator during step k
     *
     *  @return the next line m' using the line p as annihilator during step k
     *          mt if p will never be used again as an annihilator.
     */
    int (*nextpiv)(const hqr_subpiv_t *arg, int k, int p, int m);
    /*
     * prevpiv
     *    @param[in] arg pointer to the qr_piv structure
     *    @param[in] k   step in the factorization
     *    @param[in] p   line currently used as an annihilator
     *    @param[in] m   line actually annihilated.
     *          m = p to find the last time p has been used as an annihilator during step k
     *
     *  @return the previous line m' using the line p as annihilator during step k
     *          mt if p has never been used before that as an annihilator.
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
    int gmt = qrtree->mt;
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
    nb_1 += dplasma_imin( p, gmt - nb_12 );

    return dplasma_imin( nb_1 + nb_2 + nb_3, gmt - k);
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
    int gmt = qrtree->mt + 1;
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

        arg->minMN =  dplasma_imin( minMN, mt*a );
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
                nT2DO[k] = dplasma_imax( mt - ((k + p - 1 - myrank) / pa), 0 );
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
 *          HQR_LOW_GREEDY1P_TREE
 ***************************************************/
/* Return the pivot to use for the row m at step k */
inline static int hqr_low_greedy1p_currpiv( const hqr_subpiv_t *qrpiv, int k, int m ) {
    if (qrpiv->domino)
        return (qrpiv->ipiv)[ k * (qrpiv->ldd) + ( (m / qrpiv->p) / qrpiv->a ) ];
    else
        return (qrpiv->ipiv)[ ( (m%qrpiv->p) * qrpiv->minMN + k ) * (qrpiv->ldd)
                              + ( ( m  / qrpiv->p ) / qrpiv->a ) ];
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int hqr_low_greedy1p_prevpiv( const hqr_subpiv_t *qrpiv, int k, int p, int start_pa ) {
    int i;
    int p_pa = p / qrpiv->p / qrpiv->a;
    int *ipiv = qrpiv->domino ? qrpiv->ipiv : qrpiv->ipiv + p%qrpiv->p * qrpiv->minMN *qrpiv->ldd;

    for( i=start_pa+1; i<(qrpiv->ldd); i++ )
        if ( ipiv[i +  k * (qrpiv->ldd)] == p_pa )
            return i;
    return i;
 }

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int hqr_low_greedy1p_nextpiv( const hqr_subpiv_t *qrpiv, int k, int p, int start_pa ) {
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

static void hqr_low_greedy1p_init(hqr_subpiv_t *arg, int minMN){
    int *ipiv;
    int mt, a, p, pa, domino;
    int j, k, height, start, end, nT, nZ;

    arg->currpiv = hqr_low_greedy1p_currpiv;
    arg->nextpiv = hqr_low_greedy1p_nextpiv;
    arg->prevpiv = hqr_low_greedy1p_prevpiv;

    mt = arg->ldd;
    a = arg->a;
    p = arg->p;
    pa = p * a;
    domino = arg->domino;

    /* This section has not been coded yet, and will perform a classic greedy */
    if ( domino )
    {
        arg->minMN =  dplasma_imin( minMN, mt*a );
        minMN = arg->minMN;

        arg->ipiv = (int*)malloc( mt * minMN * sizeof(int) );
        ipiv = arg->ipiv;
        memset(ipiv, 0, mt*minMN*sizeof(int));

        /**
         * Compute the local greedy tree of each column, on each node
         */
        for(k=0; k<minMN; k++) {
            /* Number of tiles to factorized in this column on this rank */
            nT = dplasma_imax( mt - (k / a), 0 );
            /* Number of tiles already killed */
            nZ = 0;

            while( nZ < (nT-1) ) {
                height = (nT - nZ) / 2;
                start = mt - nZ - 1;
                end = start - height;
                nZ += height;

                for( j=start; j > end; j-- ) {
                    ipiv[ k*mt + j ] = (j - height);
                }
            }
            assert( nZ+1 == nT );
        }
    }
    else
    {
        int myrank;
        end = 0;

        arg->ipiv = (int*)malloc( mt * minMN * sizeof(int) * p );
        ipiv = arg->ipiv;

        memset( ipiv,  0, minMN*sizeof(int)*mt*p);

        for ( myrank=0; myrank<p; myrank++ ) {

            /**
             * Compute the local greedy tree of each column, on each node
             */
            for(k=0; k<minMN; k++) {
                /* Number of tiles to factorized in this column on this rank */
                nT = dplasma_imax( mt - ((k + p - 1 - myrank) / pa), 0 );
                /* Number of tiles already killed */
                nZ = 0;

                /* No more computations on this node */
                if ( nT == 0 ) {
                    break;
                }

                while( nZ < (nT-1) ) {
                    height = (nT - nZ) / 2;
                    start = mt - nZ - 1;
                    end = start - height;
                    nZ += height;

                    for( j=start; j > end; j-- ) {
                        ipiv[ myrank*mt*minMN + k*mt + j ] = (j - height);
                    }
                }
                assert( nZ+1 == nT );
            }
        }
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
            return dplasma_imin( p + arg->p - 1, arg->ldd - 1 );
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
            bit = (int)( log( (double)( dplasma_imin(arg->p, arg->ldd - k) ) ) / log( 2. ) );
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
    int end   = dplasma_imin(qrpiv->ldd-k, qrpiv->p);
    for( i=lstart+1; i<end; i++ )
        if ( (qrpiv->ipiv)[i] == lp )
            return i+k;
    return qrpiv->ldd;
}

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int hqr_high_fibonacci_nextpiv( const hqr_subpiv_t *qrpiv, int k, int p, int start ) {
    int i;
    myassert( p>=k && (start == qrpiv->ldd || start-k <= qrpiv->p) );

    for( i=dplasma_imin(start-k-1, qrpiv->p-1); i>0; i-- )
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
static void hqr_high_greedy1p_init(hqr_subpiv_t *arg)
{
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
        int j, height, start, end, firstk = 0;
        int nT = mt;
        int nZ = dplasma_imax( mt - p, 0 );

        while ( (!( ( nT == mt ) && ( nZ+1 == nT ) ) )
                && ( firstk < 1 ) )
        {
            height = (nT - nZ) / 2;
            if ( height == 0 ) {
                while ( ( firstk < 1 ) &&
                        ( nT   == mt ) &&
                        ( nZ+1 == nT ) )
                {
                    firstk++;
                }
                if (firstk > 0)
                    break;
            }

            start = mt - nZ - 1;
            end = start - height;
            nZ += height;

            for( j=start; j > end; j-- ) {
                ipiv[ j ] = (j - height);
            }
        }
    }
}

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
    for( i=dplasma_imin(start-1, k+arg->p-1); i > k; i-- )
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
        nZ[0] = dplasma_imax( mt - p, 0 );
        for(k=1; k<minMN; k++) {
            height = dplasma_imax(mt-k-p, 0);
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
static int hqr_currpiv(const dplasma_qrtree_t *qrtree, int k, int m)
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int tmp, tmpk, perm_m;
    int lm, rank, *perm;
    int a = qrtree->a;
    int p = qrtree->p;
    int domino = arg->domino;
    int gmt = qrtree->mt;

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
            /* fallthrough */
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
            /* fallthrough */
        default:
            return gmt;
        }
    }
};

/**
 *  hqr_nextpiv - Computes the next row killed by the row p, after
 *  it has kill the row start.
 *
 * @param[in] k
 *         Factorization step
 *
 * @param[in] pivot
 *         Line used as killer
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
    int gmt = qrtree->mt;

    ostart = start;
    opivot = pivot;
    start = hqr_getinvperm( qrtree, k, ostart);
    pivot = hqr_getinvperm( qrtree, k, opivot);

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
            /* fallthrough */
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
            /* fallthrough */
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
            /* fallthrough */
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
            /* fallthrough */
        case DPLASMA_QR_KILLED_BY_DISTTREE:

            if ( lp < DPLASMA_QR_KILLED_BY_DISTTREE ) {
                return gmt;
            }

            if( arg->hlvl != NULL ) {
                tmp = arg->hlvl->nextpiv( arg->hlvl, k, pivot, start );
                if ( tmp != gmt )
                    return perm[tmp];
            }
            /* fallthrough */
        default:
            return gmt;
        }
}

/**
 *  hqr_prevpiv - Computes the previous row killed by the row p, before
 *  to kill the row start.
 *
 * @param[in] k
 *         Factorization step
 *
 * @param[in] pivot
 *         Line used as killer
 *
 * @param[in] start
 *         Starting point to search the previous line killed by p before start
 *         start must be killed by p, and start must be greater or equal to p
 *
 * @return:
 *   - -1 if start doesn't respect the previous conditions
 *   -  m, the previous row killed by p if it exists, A->mt otherwise
 */
static int
hqr_prevpiv(const dplasma_qrtree_t *qrtree, int k, int pivot, int start)
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int tmp, ls, lp, nextp;
    int opivot, ostart; /* original values before permutation */
    int lpivot, rpivot, lstart, *perm;
    int a = qrtree->a;
    int p = qrtree->p;
    int gmt = qrtree->mt;

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
            /* fallthrough */
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
            /* fallthrough */
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
            /* fallthrough */
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
            /* fallthrough */
        default:
            return gmt;
        }
};

/****************************************************
 *
 * Generate the permutation required for the round-robin on TS
 *
 ***************************************************/
static void
hqr_genperm( dplasma_qrtree_t *qrtree )
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int m = qrtree->mt;
    int n = qrtree->nt;
    int a = qrtree->a;
    int p = qrtree->p;
    int domino = arg->domino;
    int minMN = dplasma_imin( m, n );
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
            end2 = dplasma_imin( end2, m );

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
            for( ; i<m; i++) {
                perm[i] = i;
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

static int
hqr_getinvperm( const dplasma_qrtree_t *qrtree, int k, int m )
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int gmt = qrtree->mt + 1;
    int a = qrtree->a;
    int p = qrtree->p;
    int pa = p * a;
    int start = m / pa * pa;
    int stop  = dplasma_imin( start + pa, gmt ) - start;
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

/**
 *******************************************************************************
 *
 * @ingroup dplasma
 *
 * dplasma_hqr_init - Creates the tree structure that will describes the
 * operation performed during QR/LQ factorization with parameterized QR/LQ
 * algorithms family.
 *
 * Trees available parameters are described below. It is recommended to:
 *   - set p to the same value than the P-by-Q process grid used to distribute
 *     the data. (P for QR factorization, Q for LQ factorization).
 *   - set the low level tree to DPLASMA_GREEDY_TREE.
 *   - set the high level tree to:
 *         1) DPLASMA_FLAT_TREE when the problem is square, because it divides
 *            by two the volume of communication of any other tree.
 *         2) DPLASMA_FIBONACCI_TREE when the problem is tall and skinny (QR) or
 *            small and fat (LQ), because it reduces the critical path length.
 *   - Disable the domino effect when problem is square, to keep high efficiency
 *     kernel proportion high.
 *   - Enable the domino effect when problem is tall and skinny (QR) or
 *     small and fat (LQ) to increase parallelism and reduce critical path length.
 *   - Round-robin on TS domain (tsrr) option should be disabled. It is
 *     experimental and is not safe.
 *
 * These are the default when a parameter is set to -1;
 *
 * See http://www.netlib.org/lapack/lawnspdf/lawn257.pdf
 *
 *******************************************************************************
 *
 * @param[in,out] qrtree
 *          On entry, an allocated structure uninitialized.
 *          On exit, the structure initialized according to the parameter given.
 *
 * @param[in] trans
 *          @arg PlasmaNoTrans:   Structure is initialized for QR factorization.
 *          @arg PlasmaTrans:     Structure is initialized for LQ factorization.
 *          @arg PlasmaConjTrans: Structure is initialized for LQ factorization.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized, on which
 *          QR/LQ factorization will be performed.
 *          The descriptor is untouched and only mt/nt/P parameters are used.
 *
 * @param[in] type_llvl
 *          Defines the tree used to reduce the main tiles of each local domain
 *          together. The matrix of those tiles has a lower triangular structure
 *          with a diagonal by step a.
 *          @arg DPLASMA_FLAT_TREE: A Flat tree is used to reduce the local
 *          tiles.
 *          @arg DPLASMA_GREEDY_TREE: A Greedy tree is used to reduce the local
 *          tiles.
 *          @arg DPLASMA_FIBONACCI_TREE: A Fibonacci tree is used to reduce the
 *          local tiles.
 *          @arg DPLASMA_BINARY_TREE: A Binary tree is used to reduce the local
 *          tiles.
 *          @arg -1: The default is used (DPLASMA_GREEDY_TREE)
 *
 * @param[in] type_hlvl
 *          Defines the tree used to reduce the main tiles of each domain. This
 *          is a band lower diagonal matrix of width p.
 *          @arg DPLASMA_FLAT_TREE: A Flat tree is used to reduce the tiles.
 *          @arg DPLASMA_GREEDY_TREE: A Greedy tree is used to reduce the tiles.
 *          @arg DPLASMA_FIBONACCI_TREE: A Fibonacci tree is used to reduce the
 *          tiles.
 *          @arg DPLASMA_BINARY_TREE: A Binary tree is used to reduce the tiles.
 *          @arg DPLASMA_GREEDY1P_TREE: A Greedy tree is computed for the first
 *          column and then duplicated on all others.
 *          @arg -1: The default is used (DPLASMA_FIBONACCI_TREE)
 *
 * @param[in] a
 *          Defines the size of the local domains on which a classic flat TS
 *          tree is performed. If a==1, then all local tiles are reduced by the
 *          type_lllvl tree. If a is larger than mt/p, then no local reduction
 *          tree is performed and type_llvl is ignored.
 *          If a == -1, it is set to 4 by default.
 *
 * @param[in] p
 *          Defines the number of distributed domains, ie the width of the high
 *          level reduction tree.  If p == 1, no high level reduction tree is
 *          used. If p == mt, a and type_llvl are ignored since only high level
 *          reduction are performed.
 *          By default, it is recommended to set p to P if trans ==
 *          PlasmaNoTrans, Q otherwise, where P-by-Q is the process grid used to
 *          distributed the data. (p > 0)
 *
 * @param[in] domino
 *          Enable/disable the domino effect that connects the high and low
 *          level reduction trees. Enabling the domino increases the proportion
 *          of TT (Triangle on top of Triangle) kernels that are less efficient,
 *          but increase the pipeline effect between independent factorization
 *          steps, reducin the critical path length.
 *          If disabled, it keeps the proprotion of more efficient TS (Triangle
 *          on top of Square) kernels high, but deteriorates the pipeline
 *          effect, and the critical path length.
 *          If domino == -1, it is enable when ration between M and N is lower
 *          than 1/2, and disabled otherwise.
 *
 * @param[in] tsrr
 *          Enable/Disable a round robin selection of the killer in local
 *          domains reduced by TS kernels. Enabling a round-robin selection of
 *          the killer allows to take benefit of the good pipelining of the flat
 *          trees and the high efficient TS kernels, while having other trees on
 *          top of it to reduce critical path length.
 *          WARNING: This option is under development and should not be enabled
 *          due to problem in corner cases.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_hqr_finalize
 * @sa dplasma_systolic_init
 * @sa dplasma_zgeqrf_param
 * @sa dplasma_cgeqrf_param
 * @sa dplasma_dgeqrf_param
 * @sa dplasma_sgeqrf_param
 *
 ******************************************************************************/
int
dplasma_hqr_init( dplasma_qrtree_t *qrtree,
                  PLASMA_enum trans, parsec_tiled_matrix_dc_t *A,
                  int type_llvl, int type_hlvl,
                  int a, int p,
                  int domino, int tsrr )
{
    double ratio = 0.0;
    int low_mt, minMN;
    hqr_args_t *arg;

    if (qrtree == NULL) {
        dplasma_error("dplasma_hqr_init", "illegal value of qrtree");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_hqr_init", "illegal value of trans");
        return -2;
    }
    if (A == NULL) {
        dplasma_error("dplasma_hqr_init", "illegal value of A");
        return -3;
    }

    /* Compute parameters */
    a = (a == -1) ? 4 : dplasma_imax( a, 1 );
    p = dplasma_imax( p, 1 );

    /* Domino */
    if ( domino >= 0 ) {
        domino = domino ? 1 : 0;
    }
    else {
        if (trans == PlasmaNoTrans) {
            ratio = ((double)(A->nt) / (double)(A->mt));
        } else {
            ratio = ((double)(A->mt) / (double)(A->nt));
        }
        if ( ratio >= 0.5 ) {
            domino = 0;
        } else {
            domino = 1;
        }
    }

    qrtree->getnbgeqrf = hqr_getnbgeqrf;
    qrtree->getm       = hqr_getm;
    qrtree->geti       = hqr_geti;
    qrtree->gettype    = hqr_gettype;
    qrtree->currpiv    = hqr_currpiv;
    qrtree->nextpiv    = hqr_nextpiv;
    qrtree->prevpiv    = hqr_prevpiv;

    qrtree->mt   = (trans == PlasmaNoTrans) ? A->mt : A->nt;
    qrtree->nt   = (trans == PlasmaNoTrans) ? A->nt : A->mt;

    a = dplasma_imin( a, qrtree->mt );

    qrtree->a    = a;
    qrtree->p    = p;
    qrtree->args = NULL;

    arg = (hqr_args_t*) malloc( sizeof(hqr_args_t) );
    arg->domino = domino;
    arg->tsrr = tsrr;
    arg->perm = NULL;

    arg->llvl = (hqr_subpiv_t*) malloc( sizeof(hqr_subpiv_t) );
    arg->hlvl = NULL;

    minMN = dplasma_imin(A->mt, A->nt);
    low_mt = (qrtree->mt + p * a - 1) / ( p * a );

    arg->llvl->minMN  = minMN;
    arg->llvl->ldd    = low_mt;
    arg->llvl->a      = a;
    arg->llvl->p      = p;
    arg->llvl->domino = domino;

    switch( type_llvl ) {
    case DPLASMA_FLAT_TREE :
        hqr_low_flat_init(arg->llvl);
        break;
    case DPLASMA_FIBONACCI_TREE :
        hqr_low_fibonacci_init(arg->llvl, minMN);
        break;
    case DPLASMA_BINARY_TREE :
        hqr_low_binary_init(arg->llvl);
        break;
    case DPLASMA_GREEDY1P_TREE :
        hqr_low_greedy1p_init(arg->llvl, minMN);
        break;
    case DPLASMA_GREEDY_TREE :
    default:
        hqr_low_greedy_init(arg->llvl, minMN);
    }

    if ( p > 1 ) {
        arg->hlvl = (hqr_subpiv_t*) malloc( sizeof(hqr_subpiv_t) );

        arg->llvl->minMN  = minMN;
        arg->hlvl->ldd    = qrtree->mt;
        arg->hlvl->a      = a;
        arg->hlvl->p      = p;
        arg->hlvl->domino = domino;

        switch( type_hlvl ) {
        case DPLASMA_FLAT_TREE :
            hqr_high_flat_init(arg->hlvl);
            break;
        case DPLASMA_GREEDY_TREE :
            hqr_high_greedy_init(arg->hlvl, minMN);
            break;
        case DPLASMA_GREEDY1P_TREE :
            hqr_high_greedy1p_init(arg->hlvl);
            break;
        case DPLASMA_BINARY_TREE :
            hqr_high_binary_init(arg->hlvl);
            break;
        case DPLASMA_FIBONACCI_TREE :
            hqr_high_fibonacci_init(arg->hlvl);
            break;
        default:
            if ( ratio >= 0.5 ) {
                hqr_high_flat_init(arg->hlvl);
            } else {
                hqr_high_fibonacci_init(arg->hlvl);
            }
        }
    }

    qrtree->args = (void*)arg;
    hqr_genperm( qrtree );

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma
 *
 * dplasma_hqr_finalize - Cleans the qrtree data structure allocated by call to
 * dplasma_hqr_init().
 *
 *******************************************************************************
 *
 * @param[in,out] qrtree
 *          On entry, an allocated structure to destroy.
 *          On exit, the structure is destroy and cannot be used.
 *
 *******************************************************************************
 *
 * @sa dplasma_hqr_init
 *
 ******************************************************************************/
void
dplasma_hqr_finalize( dplasma_qrtree_t *qrtree )
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


/*
 * Common functions
 */
static int svd_getnbgeqrf( const dplasma_qrtree_t *qrtree, int k );
static int svd_getm(       const dplasma_qrtree_t *qrtree, int k, int i   );
static int svd_geti(       const dplasma_qrtree_t *qrtree, int k, int m   );
static int svd_gettype(    const dplasma_qrtree_t *qrtree, int k, int m   );

#define svd_getipiv( __qrtree, _k ) ((__qrtree)->llvl->ipiv + ((__qrtree)->llvl->ldd) * (_k) )
#define svd_geta( __qrtree, _k ) ( (svd_getipiv( (__qrtree), (_k) ))[0] )

/*
 * Extra parameter:
 *    gmt - Global number of tiles in a column of the complete distributed matrix
 * Return:
 *    The number of geqrt to execute in the panel k
 */
static int
svd_getnbgeqrf( const dplasma_qrtree_t *qrtree,
                int k )
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int p = qrtree->p;
    int gmt = qrtree->mt;
    int a = svd_geta(arg, k);
    int pa = p * a;
    int nb_1, nb_2, nb_3;
    int nb_11, nb_12;

    /* Number of tasks of type 3 */
    nb_3 = p;

    /* Number of extra tile of type 1 between the tile of type 3 and the first of nb11 */
    nb_2 = nbextra1_formula;

    /* First multiple of p*a under the diagonal of step 1 */
    nb_11 = ( (k + p + pa - 1 ) / pa ) * pa;

    /* Last multiple of p*a lower than A->mt */
    nb_12 = ( gmt / pa ) * pa;

    /* Number of tasks of type 1 between nb_11 and nb_12 */
    nb_1 = (nb_12 - nb_11) / a;

    /* Add leftover */
    nb_1 += dplasma_imin( p, gmt - nb_12 );

    return dplasma_imin( nb_1 + nb_2 + nb_3, gmt - k);
}

/*
 * Extra parameter:
 *    i - indice of the geqrt in the continuous space
 * Return:
 *    The global indice m of the i th geqrt in the panel k
 */
static int
svd_getm( const dplasma_qrtree_t *qrtree,
          int k, int i )
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int p = qrtree->p;
    int a = svd_geta(arg, k);

    int pos1, j, pa = p * a;
    int nbextra1 = nbextra1_formula;
    int nb23 = p + nbextra1;

    /* Tile of type 2 or 3 or the 1 between the diagonal and the multiple after the diagonal */
    if ( i < nb23 )
        return k+i;
    /* Tile of type 1 */
    else {
        j = i - nb23;
        pa = p * a;
        pos1 = ( ( (p + k    ) + pa - 1 ) / pa ) * pa;
        return pos1 + (j/p) * pa + j%p;
    }
}

/*
 * Extra parameter:
 *    m - The global indice m of a geqrt in the panel k
 * Return:
 *    The index i of the geqrt in the panel k
 */
static int
svd_geti( const dplasma_qrtree_t *qrtree,
          int k, int m )
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int p = qrtree->p;
    int a = svd_geta(arg, k);

    int pos1, j, pa = p * a;
    int nbextra1 = nbextra1_formula;
    int nb23 = p + nbextra1;
    int end2 = p + k + nbextra1;

    /* Tile of type 2 or 3 or the 1 between the diagonal and the multiple after the diagonal */
    if ( m < end2 )
        return m-k;
    /* Tile of type 1 */
    else {
        pos1 = ( ( (p + k    ) + pa - 1 ) / pa ) * pa;
        j = m - pos1;
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
static int
svd_gettype( const dplasma_qrtree_t *qrtree,
             int k, int m )
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int p = qrtree->p;
    int a = svd_geta(arg, k);

    /* Element to be reduce in distributed */
    if (m < k + p) {
        return 3;
    }
    /* Lower triangle of the matrix */
    else {
        if( (m / p) % a == 0 )
            return 1;
        else
            return 0;
    }
}

/****************************************************
 *          SVD_LOW_ADAPTIV_TREE
 ***************************************************/
/* Return the pivot to use for the row m at step k */
inline static int
svd_low_adaptiv_currpiv( const hqr_subpiv_t *arg,
                         int k, int m )
{
    int *ipiv = arg->ipiv + (m%arg->p * arg->minMN + k) * arg->ldd;
    int a = ipiv[0];

    ipiv+=2;
    return ipiv[ ( m  / arg->p ) / a ];
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int
svd_low_adaptiv_prevpiv( const hqr_subpiv_t *arg,
                         int k, int p, int start_pa )
{
    int i;
    int *ipiv = arg->ipiv + (p%arg->p * arg->minMN + k) * arg->ldd;
    int a = ipiv[0];
    int ldd = ipiv[1];
    int p_pa = p / arg->p / a;

    ipiv+=2;
    for( i=start_pa+1; i<ldd; i++ )
        if ( ipiv[i] == p_pa )
            return i;
    return i;
}

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int
svd_low_adaptiv_nextpiv( const hqr_subpiv_t *arg,
                         int k, int p, int start_pa )
{
    int i;
    int *ipiv = arg->ipiv + (p%arg->p * arg->minMN + k ) * arg->ldd;
    int a = ipiv[0];
    int ldd = ipiv[1];
    int pa = arg->p * a;
    int k_a = (k + arg->p - 1 - p%(arg->p)) / arg->p / a;
    int p_pa = p / pa;

    ipiv+=2;
    for( i=start_pa-1; i> k_a; i-- )
        if ( ipiv[i] == p_pa )
            return i;

    return ldd;
}

static void
svd_low_adaptiv_init(hqr_subpiv_t *arg,
                     int gmt, int gnt, int nbcores, int ratio)
{
    int *ipiv;
    int mt, a, p, pa, maxmt, myrank;
    int j, k, height, start, end, nT, nZ;
    int minMN = dplasma_imin(gmt, gnt);

    arg->currpiv = svd_low_adaptiv_currpiv;
    arg->nextpiv = svd_low_adaptiv_nextpiv;
    arg->prevpiv = svd_low_adaptiv_prevpiv;

    p = arg->p;

    end = 0;

    /**
     * Compute the local greedy tree of each column, on each node
     */
    maxmt = 1;
    for(k=0; k<minMN; k++) {
        /**
         * The objective is to have at least two columns of TS to reduce per
         * core, so it must answer the following inequality:
         * ((gmt-k) / p / a ) * (gnt-k) >= ( ratio * nbcores );
         * so,
         * a <= mt * (gnt-k) / (ratio * nbcores )
         */
        height = dplasma_iceil( gmt-k, p );
        a = dplasma_imax( height * (gnt-k) / (ratio * nbcores), 1 );

        /* Now let's make sure all sub-parts are equilibrate */
        j = dplasma_iceil( height, a );
        a = dplasma_iceil( gmt-k, j );

        /* Compute max dimension of the tree */
        mt = dplasma_iceil( gmt, p * a );
        maxmt = dplasma_imax( mt, maxmt );
    }

    arg->ldd = maxmt + 2;
    arg->ipiv = (int*)malloc( arg->ldd * minMN * sizeof(int) * p );
    ipiv = arg->ipiv;

    memset( ipiv, 0, minMN*sizeof(int)*arg->ldd*p );

    for ( myrank=0; myrank<p; myrank++ ) {

        /**
         * Compute the local greedy tree of each column, on each node
         */
        for(k=0; k<minMN; k++, ipiv += arg->ldd) {
            /**
             * The objective is to have at least two columns of TS to reduce per
             * core, so it must answer the following inequality:
             * (ldd / a ) * (gnt-k) >= ( ratio * nbcores );
             * so,
             * a <= mt * (gnt-k) / (ratio * nbcores )
             */
            height = dplasma_iceil( gmt-k, p );
            a = dplasma_imax( height * (gnt-k) / (ratio * nbcores), 1 );

            /* Now let's make sure all sub-parts are equilibrate */
            j = dplasma_iceil( height, a );
            a = dplasma_iceil( gmt-k, j );

            pa = p * a;
            mt = dplasma_iceil( gmt, pa );
            ipiv[0] = a;
            ipiv[1] = mt;

            assert( a  > 0 );
            assert( mt < arg->ldd-1 );

            /* Number of tiles to factorized in this column on this rank */
            nT = dplasma_imax( mt - ((k + p - 1 - myrank) / pa), 0 );
            /* Number of tiles already killed */
            nZ = 0;

            assert( nT <= mt );

            /* No more computations on this node */
            if ( nT == 0 ) {
                continue;
            }

            while( nZ < (nT-1) ) {
                height = (nT - nZ) / 2;
                start = mt - nZ - 1;
                end = start - height;
                nZ += height;

                for( j=start; j > end; j-- ) {
                    ipiv[ j+2 ] = (j - height);
                }
            }
            assert( nZ+1 == nT );
        }
    }

#if 0
    {
        int m, k;
        for(m=0; m<mt; m++) {
            printf("%3d | ", m);
            for (k=0; k<minMN; k++) {
                printf( "%3d ", ipiv[k*(arg->ldd) + m] );
            }
            printf("\n");
        }
    }
    if (!arg->domino) {
        int m, k, myrank;
        for ( myrank=1; myrank<p; myrank++ ) {
            ipiv += arg->ldd * minMN;
            printf("-------- rank %d ---------\n", myrank );
            for(m=0; m<mt; m++) {
                printf("%3d | ", m);
                for (k=0; k<minMN; k++) {
                    int k_a = (k + p - 1 - myrank) / p / a;
                    if ( m >= k_a )
                        printf( "%3d ", ipiv[k * arg->ldd + m] );
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
 *
 *   Generic functions currpiv,prevpiv,nextpiv
 *
 ***************************************************/
static int svd_currpiv(const dplasma_qrtree_t *qrtree, int k, int m)
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int tmp, tmpk;
    int lm, rank;
    int a = svd_geta( arg, k );
    int p = qrtree->p;
    int gmt = qrtree->mt;

    lm   = m / p; /* Local index in the distribution over p domains */
    rank = m % p; /* Staring index in this distribution             */

    /* TS level common to every case */
    switch( svd_gettype( qrtree, k, m ) )
    {
    case 0:
        tmp = lm / a;
        /* tmpk = (k + p - 1 - m%p) / p / a;  */
        tmpk = k / (p * a);
        return ( tmp == tmpk ) ? k + (m-k)%p : tmp * a * p + rank;
        break;
    case 1:
        tmp = arg->llvl->currpiv(arg->llvl, k, m);
        /* tmpk = (k + p - 1 - m%p) / p / a; */
        tmpk = k / (p * a);
        return ( tmp == tmpk ) ? k + (m-k)%p : tmp * a * p + rank;
        break;
    case 2:
        assert(0);
        break;
    case 3:
        if ( arg->hlvl != NULL )
            return arg->hlvl->currpiv(arg->hlvl, k, m);
        /* fallthrough */
    default:
        return gmt;
    }
    return -1;
};

/**
 *  svd_nextpiv - Computes the next row killed by the row p, after
 *  it has kill the row start.
 *
 * @param[in] k
 *         Factorization step
 *
 * @param[in] pivot
 *         Line used as killer
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
static int svd_nextpiv(const dplasma_qrtree_t *qrtree, int k, int pivot, int start)
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int tmp, ls, lp, nextp;
    int rpivot, lstart;
    int *ipiv = svd_getipiv( arg, k );
    int a = ipiv[0];
    int ldd = ipiv[1];
    int p = qrtree->p;
    int gmt = qrtree->mt;

    rpivot = pivot % p; /* Staring index in this distribution             */

    /* Local index in the distribution over p domains */
    lstart = ( start == gmt ) ? ldd * a : start / p;

    myassert( start > pivot && pivot >= k );
    myassert( start == gmt || pivot == svd_currpiv( qrtree, k, start ) );

    /* TS level common to every case */
    ls = (start < gmt) ? svd_gettype( qrtree, k, start ) : -1;
    lp = svd_gettype( qrtree, k, pivot );

    switch( ls )
        {
        case DPLASMA_QR_KILLED_BY_DOMINO:
            assert(0);

        case -1:

            if ( lp == DPLASMA_QR_KILLED_BY_TS ) {
                myassert( start == gmt );
                return gmt;
            }
            /* fallthrough */
        case DPLASMA_QR_KILLED_BY_TS:
            if ( start == gmt )
                nextp = pivot + p;
            else
                nextp = start + p;

            if ( ( nextp < gmt ) &&
                 ( nextp < pivot + a*p ) &&
                 ( (nextp/p)%a != 0 ) )
                return nextp;
            start = gmt;
            lstart = ldd * a;
            /* fallthrough */
        case DPLASMA_QR_KILLED_BY_LOCALTREE:
            /* Get the next pivot for the low level tree */
            tmp = arg->llvl->nextpiv(arg->llvl, k, pivot, lstart / a );

            if ( (tmp * a * p + rpivot >= gmt)
                 && (tmp == ldd-1) )
                tmp = arg->llvl->nextpiv(arg->llvl, k, pivot, tmp);

            if ( tmp != ldd )
                return tmp * a * p + rpivot;

            /* no next of type 1, we reset start to search the next 2 */
            start = gmt;
            lstart = ldd * a;
            /* fallthrough */
        case DPLASMA_QR_KILLED_BY_DISTTREE:

            if ( lp < DPLASMA_QR_KILLED_BY_DISTTREE ) {
                return gmt;
            }

            if( arg->hlvl != NULL ) {
                tmp = arg->hlvl->nextpiv( arg->hlvl, k, pivot, start );
                if ( tmp != gmt )
                    return tmp;
            }
            /* fallthrough */
        default:
            return gmt;
        }
}

/**
 *  svd_prevpiv - Computes the previous row killed by the row p, before
 *  to kill the row start.
 *
 * @param[in] k
 *         Factorization step
 *
 * @param[in] pivot
 *         Line used as killer
 *
 * @param[in] start
 *         Starting point to search the previous line killed by p before start
 *         start must be killed by p, and start must be greater or equal to p
 *
 * @return:
 *   - -1 if start doesn't respect the previous conditions
 *   -  m, the previous row killed by p if it exists, A->mt otherwise
 */
static int
svd_prevpiv(const dplasma_qrtree_t *qrtree, int k, int pivot, int start)
{
    hqr_args_t *arg = (hqr_args_t*)(qrtree->args);
    int tmp, ls, lp, nextp;
    int lpivot, rpivot, lstart;
    int *ipiv = svd_getipiv( arg, k );
    int a = ipiv[0];
    int ldd = ipiv[1];
    int p = qrtree->p;
    int gmt = qrtree->mt;

    lpivot = pivot / p; /* Local index in the distribution over p domains */
    rpivot = pivot % p; /* Staring index in this distribution             */
    lstart = start / p; /* Local index in the distribution over p domains */

    myassert( start >= pivot && pivot >= k && start < gmt );
    myassert( start == pivot || pivot == svd_currpiv( qrtree, k, start ) );

    /* TS level common to every case */
    ls = svd_gettype( qrtree, k, start );
    lp = svd_gettype( qrtree, k, pivot );

    if ( lp == DPLASMA_QR_KILLED_BY_TS )
      return gmt;

    myassert( lp >= ls );
    switch( ls )
        {
        case DPLASMA_QR_KILLED_BY_DOMINO:
            assert(0);
        case DPLASMA_QR_KILLED_BY_DISTTREE:
            if( arg->hlvl != NULL ) {
                tmp = arg->hlvl->prevpiv( arg->hlvl, k, pivot, start );
                if ( tmp != gmt )
                    return tmp;
            }

            start = pivot;
            lstart = pivot / p;
            /* fallthrough */
        case DPLASMA_QR_KILLED_BY_LOCALTREE:
            tmp = arg->llvl->prevpiv(arg->llvl, k, pivot, lstart / a);

            if ( (tmp * a * p + rpivot >= gmt)
                 && (tmp == ldd-1) )
                tmp = arg->llvl->prevpiv(arg->llvl, k, pivot, tmp);

            if ( tmp != ldd )
                return tmp * a * p + rpivot;

            start = pivot;
            /* fallthrough */
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
                return nextp;
            /* fallthrough */
        default:
            return gmt;
        }
};

/**
 *******************************************************************************
 *
 * @ingroup dplasma
 *
 * dplasma_svd_init - Create the tree structures that will describes the
 * operation performed during QR/LQ reduction step of the gebrd_ge2gb operation.
 *
 * Trees available parameters are described below. It is recommended to:
 *   - set p to the same value than the P-by-Q process grid used to distribute
 *     the data. (P for QR factorization, Q for LQ factorization).
 *   - set the low level tree to DPLASMA_GREEDY_TREE.
 *   - set the high level tree to:
 *         1) DPLASMA_FLAT_TREE when the problem is square, because it divides
 *            by two the volume of communication of any other tree.
 *         2) DPLASMA_FIBONACCI_TREE when the problem is tall and skinny (QR) or
 *            small and fat (LQ), because it reduces the critical path length.
 *   - Disable the domino effect when problem is square, to keep high efficiency
 *     kernel proportion high.
 *   - Enable the domino effect when problem is tall and skinny (QR) or
 *     small and fat (LQ) to increase parallelism and reduce critical path length.
 *   - Round-robin on TS domain (tsrr) option should be disabled. It is
 *     experimental and is not safe.
 *
 * These are the default when a parameter is set to -1;
 *
 * See http://www.netlib.org/lapack/lawnspdf/lawn257.pdf
 *
 *******************************************************************************
 *
 * @param[in,out] qrtree
 *          On entry, an allocated structure uninitialized.
 *          On exit, the structure initialized according to the given parameters.
 *
 * @param[in] trans
 *          @arg PlasmaNoTrans:   Structure is initialized for the QR steps.
 *          @arg PlasmaTrans:     Structure is initialized for the LQ steps.
 *          @arg PlasmaConjTrans: Structure is initialized for the LQ steps.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized, on which
 *          QR/LQ reduction steps will be performed. In case, of
 *          R-bidiagonalization, don't forget to provide the square submatrix
 *          that is concerned by those operations.
 *          The descriptor is untouched and only mt/nt/P parameters are used.
 *
 * @param[in] type_hlvl
 *          Defines the tree used to reduce the main tiles of each domain. This
 *          is a band lower diagonal matrix of width p.
 *          @arg DPLASMA_FLAT_TREE: A Flat tree is used to reduce the tiles.
 *          @arg DPLASMA_GREEDY_TREE: A Greedy tree is used to reduce the tiles.
 *          @arg DPLASMA_FIBONACCI_TREE: A Fibonacci tree is used to reduce the
 *          tiles.
 *          @arg DPLASMA_BINARY_TREE: A Binary tree is used to reduce the tiles.
 *          @arg DPLASMA_GREEDY1P_TREE: A Greedy tree is computed for the first
 *          column and then duplicated on all others.
 *          @arg -1: The default is used (DPLASMA_FIBONACCI_TREE)
 *
 * @param[in] p
 *          Defines the number of distributed domains, ie the width of the high
 *          level reduction tree.  If p == 1, no high level reduction tree is
 *          used. If p == mt, this enforce the high level reduction tree to be
 *          performed on the full matrix.
 *          By default, it is recommended to set p to P if trans ==
 *          PlasmaNoTrans, and to Q otherwise, where P-by-Q is the process grid
 *          used to distributed the data. (p > 0)
 *
 * @param[in] nbthread_per_node
 *          Define the number of working threads per node to configure the
 *          adaptativ local tree to provide at least (ratio * nbthread_per_node)
 *          tasks per step when possible by creating the right amount of TS and
 *          TT kernels.
 *
 * @param[in] ratio
 *          Define the minimal number of tasks per thread that the adaptiv tree
 *          must provide at the lowest level of the tree.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_hqr_finalize
 * @sa dplasma_hqr_init
 * @sa dplasma_zgeqrf_param
 * @sa dplasma_cgeqrf_param
 * @sa dplasma_dgeqrf_param
 * @sa dplasma_sgeqrf_param
 *
 ******************************************************************************/
int
dplasma_svd_init( dplasma_qrtree_t *qrtree,
                  PLASMA_enum trans, parsec_tiled_matrix_dc_t *A,
                  int type_hlvl, int p, int nbthread_per_node, int ratio )
{
    int low_mt, minMN, a = -1;
    hqr_args_t *arg;

    if (qrtree == NULL) {
        dplasma_error("dplasma_svd_init", "illegal value of qrtree");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_svd_init", "illegal value of trans");
        return -2;
    }
    if (A == NULL) {
        dplasma_error("dplasma_svd_init", "illegal value of A");
        return -3;
    }

    /* Compute parameters */
    p = dplasma_imax( p, 1 );

    qrtree->getnbgeqrf = svd_getnbgeqrf;
    qrtree->getm       = svd_getm;
    qrtree->geti       = svd_geti;
    qrtree->gettype    = svd_gettype;
    qrtree->currpiv    = svd_currpiv;
    qrtree->nextpiv    = svd_nextpiv;
    qrtree->prevpiv    = svd_prevpiv;

    qrtree->mt   = (trans == PlasmaNoTrans) ? A->mt : A->nt;
    qrtree->nt   = (trans == PlasmaNoTrans) ? A->nt : A->mt;

    qrtree->a    = a;
    qrtree->p    = p;
    qrtree->args = NULL;

    arg = (hqr_args_t*) malloc( sizeof(hqr_args_t) );
    arg->domino = 0;
    arg->tsrr = 0;
    arg->perm = NULL;

    arg->llvl = (hqr_subpiv_t*) malloc( sizeof(hqr_subpiv_t) );
    arg->hlvl = NULL;

    minMN = dplasma_imin(A->mt, A->nt);
    low_mt = (qrtree->mt + p - 1) / ( p );

    arg->llvl->minMN  = minMN;
    arg->llvl->ldd    = low_mt;
    arg->llvl->a      = a;
    arg->llvl->p      = p;
    arg->llvl->domino = 0;

    svd_low_adaptiv_init(arg->llvl, qrtree->mt, qrtree->nt,
                         nbthread_per_node * (A->super.nodes / p), ratio );

    if ( p > 1 ) {
        arg->hlvl = (hqr_subpiv_t*) malloc( sizeof(hqr_subpiv_t) );

        arg->llvl->minMN  = minMN;
        arg->hlvl->ldd    = qrtree->mt;
        arg->hlvl->a      = a;
        arg->hlvl->p      = p;
        arg->hlvl->domino = 0;

        switch( type_hlvl ) {
        case DPLASMA_FLAT_TREE :
            hqr_high_flat_init(arg->hlvl);
            break;
        case DPLASMA_GREEDY_TREE :
            hqr_high_greedy_init(arg->hlvl, minMN);
            break;
        case DPLASMA_GREEDY1P_TREE :
            hqr_high_greedy1p_init(arg->hlvl);
            break;
        case DPLASMA_BINARY_TREE :
            hqr_high_binary_init(arg->hlvl);
            break;
        case DPLASMA_FIBONACCI_TREE :
            hqr_high_fibonacci_init(arg->hlvl);
            break;
        default:
            hqr_high_fibonacci_init(arg->hlvl);
        }
    }

    qrtree->args = (void*)arg;

    return 0;
}
