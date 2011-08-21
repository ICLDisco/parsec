/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include <math.h>
#include <plasma.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"
#include "pivgen.h"

#ifndef min
#define min(__a, __b) ( ( (__a) < (__b) ) ? (__a) : (__b) )
#endif

#ifndef max
#define max(__a, __b) ( ( (__a) > (__b) ) ? (__a) : (__b) )
#endif

#define PIV(_ipiv, _mt, _i, _k) (_piv)[(_k)*(_mt)+(_i)]
/* #define PRINT_PIVGEN 1 */

#ifdef PRINT_PIVGEN
#define myassert( test ) if ( ! (test) ) return -1;
#else
#define myassert assert
#endif

/*
 * Common functions
 */
int dplasma_qr_getnbgeqrf( const int a, const int p, const int k, const int gmt );
int dplasma_qr_getm(       const int a, const int p, const int k, const int i   );
int dplasma_qr_geti(       const int a, const int p, const int k, const int m   );
int dplasma_qr_gettype(    const int a, const int p, const int k, const int m   );

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
void dplasma_flat_init(qr_subpiv_t *arg, const int mt, const int a);
int  dplasma_flat_currpiv(const qr_subpiv_t *arg, const int m, const int k);
int  dplasma_flat_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start);
int  dplasma_flat_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start);

void dplasma_binary_init(qr_subpiv_t *arg, const int mt, const int a);
int  dplasma_binary_currpiv(const qr_subpiv_t *arg, const int m, const int k);
int  dplasma_binary_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start);
int  dplasma_binary_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start);

void dplasma_fibonacci_init(qr_subpiv_t *arg, const int mt, const int minMN, const int a);
void dplasma_greedy_init(   qr_subpiv_t *arg, const int mt, const int minMN, const int a);


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

/*
 * Extra parameter:
 *    gmt - Global number of tiles in a column of the complete distributed matrix
 * Return:
 *    The number of geqrt to execute in the panel k
 */
int dplasma_qr_getnbgeqrf( const int a, const int p, const int k, const int gmt ) {
    int pa = p * a;
    int nb_1, nb_2, nb_3;
    int nb_11, nb_12;

    /* Number of tasks of type 3 */
    nb_3 = p;
    
    /* Number of tasks of type 2 */
    nb_2 = k * (p - 1);

    /* First multiple of p*a under the diagonal */
    nb_11 =  ( ( (p * (k+1)) + pa - 1 ) / pa ) * pa;
    
    /* Last multiple of p*a lower than A->mt */
    nb_12 =  ( gmt / pa ) * pa;

    /* Number of tasks of type 1 between nb_11 and nb_12 */
    nb_1 = (nb_12 - nb_11) / a;
    
    /* Add leftover */
    nb_1 += min( p, gmt - nb_12 );
    
    /* nb = gmt - p * (k+1) - ( (p * (k+1))%a )*p; */
    /* nb = max( nb, 0 ); */
    /* nb = ( nb / ( p * a ) ) * p + nb_2 + nb_1 + (( gmt - p*(k+1) < gmt%(p*a) ) ? 0 : min( gmt % (p*a), p )); */

    return min( nb_1 + nb_2 + nb_3, gmt - k);
}

/*
 * Extra parameter:
 *    i - indice of the geqrt in the continuous space
 * Return:
 *    The global indice m of the i th geqrt in the panel k
 */
int dplasma_qr_getm( const int a, const int p, const int k, const int i)
{
    int pos1, j, pa;
    int nb23 = p + k*(p-1);
    
    /* Tile of type 2 or 3 */
    if ( i < nb23 )
        return k+i;
    /* Tile of type 1 */
    else {
        pa = p * a;
        pos1 = ( ( (p * (k+1)) + pa - 1 ) / pa ) * pa;
        j = i - nb23;
        return pos1 + (j/p) * a * p + j%p ;
    }
}

/*
 * Extra parameter:
 *    m - The global indice m of a geqrt in the panel k
 * Return:
 *    The index i of the geqrt in the panel k 
 */
int dplasma_qr_geti( const int a, const int p, const int k, const int m)
{
    int pos1, j, pa;
    int nb23 = p + k*(p-1);
    int end2 = (k+1)*p;
    
    /* Tile of type 2 or 3 */
    if ( m < end2 )
        return m-k;
    /* Tile of type 1 */
    else {
        pa = p * a;
        pos1 = ( ( (p * (k+1)) + pa - 1 ) / pa ) * pa;
        j = m - pos1;
        return nb23 + (j / (p*a)) * p + j%(p*a) ;
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
int dplasma_qr_gettype( const int a, const int p, const int k, const int m ) {
    myassert( m >= k );

    /* Element to be reduce in distributed */
    if (m < k + p) {
        return 3;
    }
    /* Lower triangle of the matrix */
    else if ( ( m / p) > k ) {
        if( (m / p) % a == 0 )
            return 1;
        else
            return 0;
    }
    /* Element on the local diagonal */
    else {
        return 2;
    } 
}

/****************************************************
 *                 DPLASMA_FLAT_TREE
 ***************************************************/
int dplasma_flat_currpiv(const qr_subpiv_t *arg, const int m, const int k) 
{ 
    (void)m;
    return k / arg->a;
};

int dplasma_flat_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{ 
#ifdef FLAT_UP
    if ( ( p == (k/arg->a) ) && (start > (k/arg->a)+1 ) )
        return start-1;
    else
#else /* FLAT_DOWN */
    if ( p == (k/arg->a) && ( arg->ldd - k/arg->a ) > 1 ) {
        if ( start == arg->ldd )
            return p+1;
        else if ( start < arg->ldd )
            return start+1;
    }
#endif
    return arg->ldd;
};

int dplasma_flat_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
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

void dplasma_flat_init(qr_subpiv_t *arg, int mt, int a){
    arg->currpiv = dplasma_flat_currpiv;
    arg->nextpiv = dplasma_flat_nextpiv;
    arg->prevpiv = dplasma_flat_prevpiv;
    arg->ipiv = NULL;
    arg->ldd  = mt;
    arg->a    = a;
};

/****************************************************
 *                 DPLASMA_BINARY_TREE
 ***************************************************/
int dplasma_binary_currpiv(const qr_subpiv_t *arg, const int m, const int k) 
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

int dplasma_binary_nextpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{ 
    int tmpp, bit;
    int lk = (k / arg->a);

    myassert( (start == arg->ldd) || (dplasma_binary_currpiv( arg, start, k ) == p) );

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

int dplasma_binary_prevpiv(const qr_subpiv_t *arg, const int p, const int k, const int start)
{ 
    int lk = (k / arg->a);
    myassert( start >= p && ( start == p || dplasma_binary_currpiv( arg, start, k ) == p) );

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

void dplasma_binary_init(qr_subpiv_t *arg, int mt, int a){
    arg->currpiv = dplasma_binary_currpiv;
    arg->nextpiv = dplasma_binary_nextpiv;
    arg->prevpiv = dplasma_binary_prevpiv;
    arg->ipiv = NULL;
    arg->ldd  = mt;
    arg->a    = a;
};

/****************************************************
 *          DPLASMA_FIBONACCI_TREE
 ***************************************************/
/* Return the pivot to use for the row m at step k */
inline static int dplasma_fibonacci_currpiv( const qr_subpiv_t *qrpiv, const int m, const int k ) {
    return (qrpiv->ipiv)[ (k/qrpiv->a) * (qrpiv->ldd) + m ];
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int dplasma_fibonacci_prevpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    for( i=start+1; i<(qrpiv->ldd); i++ )
        if ( (qrpiv->ipiv)[i +  (k/qrpiv->a) * (qrpiv->ldd)] == p )
            return i;
    return i;
 }

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int dplasma_fibonacci_nextpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    for( i=start-1; i>(k/qrpiv->a); i-- )
        if ( (qrpiv->ipiv)[i + (k/qrpiv->a) * (qrpiv->ldd)] == p )
            return i;
    return (qrpiv->ldd);
}

void dplasma_fibonacci_init(qr_subpiv_t *arg, int mt, int minMN, int a){
    int *ipiv;

    arg->currpiv = dplasma_fibonacci_currpiv;
    arg->nextpiv = dplasma_fibonacci_nextpiv;
    arg->prevpiv = dplasma_fibonacci_prevpiv;
    arg->ipiv = (int*)malloc( mt * minMN * sizeof(int) );
    arg->ldd  = mt;
    arg->a    = a;
    ipiv = arg->ipiv;
    memset(ipiv, 0, mt*minMN*sizeof(int));
   
    {
        int f0, f1, f2, k, m;
        
        /* Fill in the first column */
        f0 = 0;
        f1 = 1;
        for (m=1; m < mt; ) {
            for (k=0; (k < f1) && (m < mt); k++, m++) {
                ipiv[m] = m - f1;
            }
            f2 = f0 + f1;
            f0 = f1;
            f1 = f2;
        }

        for( k=1; k<minMN; k++) {
            for(m=k+1; m < mt; m++) {
                ipiv[ k * mt + m ] = ipiv[ (k-1) * mt + m - 1 ] + 1;
            }
        }
    }
};

/****************************************************
 *          DPLASMA_GREEDY_TREE
 ***************************************************/
/* Return the pivot to use for the row m at step k */
inline static int currpiv( const qr_subpiv_t *qrpiv, const int m, const int k ) {
    return (qrpiv->ipiv)[ k * (qrpiv->ldd) + m ];
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int prevpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    for( i=start+1; i<(qrpiv->ldd); i++ )
        if ( (qrpiv->ipiv)[i +  k * (qrpiv->ldd)] == p )
            return i;
    return i;
 }

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int nextpiv( const qr_subpiv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    for( i=start-1; i>(k/qrpiv->a); i-- )
        if ( (qrpiv->ipiv)[i + k* (qrpiv->ldd)] == p )
            return i;
    return (qrpiv->ldd);
}

void dplasma_greedy_init(qr_subpiv_t *arg, int mt, int minMN, int a){
    int *ipiv;

    arg->currpiv = currpiv;
    arg->nextpiv = nextpiv;
    arg->prevpiv = prevpiv;
    arg->ipiv = (int*)malloc( mt * minMN * sizeof(int) );
    arg->ldd  = mt;
    arg->a    = a;
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
        while ( ! ( ( nT[minMN-1] == mt - ( (minMN - 1) / a ) ) &&
                    ( nZ[minMN-1]+1 == nT[minMN-1] ) ) ) {
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
 *       DPLASMA_FLAT_TREE / DPLASMA_FLAT_TREE
 ***************************************************/
static int dplasma_qr_currpiv(const qr_piv_t *arg, const int m, const int k) 
{ 
    int tmp;
    int a    = arg->a;
    int p    = arg->p;
    int lm   = m / p; /* Local index in the distribution over p domains */
    int rank = m % p; /* Staring index in this distribution             */

    /* TS level common to every case */
    switch( dplasma_qr_gettype( a, p, k, m ) ) 
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
            /* tmp = arg->highlevel->currpiv(arg, lm / a, k); */
            /* return ( (lm / a) == tmp ) ? k*p+rank : tmp * a * p + rank;  */
            return -2;
            break;
        default:
            return arg->desc->mt;
        }
};

static int dplasma_qr_nextpiv(const qr_piv_t *arg, const int pivot, const int k, const int start)
{ 
    int tmp, ls, lp, nextp;
    int a    = arg->a;
    int p    = arg->p;
    int lpivot = pivot / p; /* Local index in the distribution over p domains */
    int rpivot = pivot % p; /* Staring index in this distribution             */
    int st = ( start == arg->desc->mt ) ? arg->desc->mt + p - 1 : start ;
    int lstart = st / p; /* Local index in the distribution over p domains */

    myassert( start > pivot && pivot >= k );
    myassert( start == arg->desc->mt || pivot == dplasma_qr_currpiv( arg, start, k ) );
            
    /* TS level common to every case */
    ls = (start < arg->desc->mt) ? dplasma_qr_gettype( a, p, k, start ) : -1;
    lp = dplasma_qr_gettype( a, p, k, pivot );

    if ( lp == 0 ) {
        myassert( start == arg->desc->mt );
        return arg->desc->mt;
    }

    /* First query / Check for use in TS */
    if ( start == arg->desc->mt || ls == 0 ) {
        tmp = min( lstart - 1, lpivot + a - 1 - lpivot%a );
        nextp = tmp * p + rpivot;
 
        if ( pivot < nextp ) 
            return nextp; 

        tmp = arg->llvl->nextpiv(arg->llvl, lpivot / a, k, arg->llvl->ldd);
    } 
    else {
        /* Get the next pivot for the low level tree */
        tmp = arg->llvl->nextpiv(arg->llvl, lpivot / a, k, (lstart+a-1) / a);
    }

    if ( tmp != arg->llvl->ldd )
        return tmp * a * p + rpivot;
    else if ( lp == 1 || ( lp == 2 && lpivot == k) )
        return arg->desc->mt;

    if ( (start == arg->desc->mt) && 
         (lpivot < k)             &&
         (pivot+p < arg->desc->mt) ) {
        return pivot+p;
    }

    myassert( lp == 3 );
    if( arg->hlvl == NULL ) {
        if ( start == pivot + p * a )
            return arg->desc->mt;
    } else {
        tmp = arg->hlvl->nextpiv( arg->hlvl, lpivot / a, k, lstart / a );
        if ( tmp != arg->hlvl->ldd )
            return tmp * a * p + rpivot; /* ToDo: fix */
    }
    return arg->desc->mt;
}

static int dplasma_qr_prevpiv(const qr_piv_t *arg, const int pivot, const int k, const int start)
{ 
    int tmp, ls;
    int a    = arg->a;
    int p    = arg->p;
    int lpivot = pivot / p; /* Local index in the distribution over p domains */
    int rpivot = pivot % p; /* Staring index in this distribution             */
    int lstart = start / p; /* Local index in the distribution over p domains */
    int startts = start;

    myassert( start >= pivot && pivot >= k && start < arg->desc->mt );
    myassert( start == pivot || pivot == dplasma_qr_currpiv( arg, start, k ) );
            
    /* TS level common to every case */
    ls = dplasma_qr_gettype( a, p, k, start );
    switch( dplasma_qr_gettype( a, p, k, pivot ) )
        {
        case 0:
            return arg->desc->mt;
            break;

        case 3:
            if( arg->hlvl != NULL ) {
                tmp = arg->hlvl->prevpiv( arg->hlvl, lpivot / a, k, lstart / a );
                if ( tmp != arg->hlvl->ldd )
                    return tmp * a * p + rpivot; /* ToDo: fix */
            }

        case 2:
            /* If inside the band */
            if ( ( lpivot < k ) && ( lpivot > (k+1+rpivot) ) ) {
                if ( (start == pivot) && (start+p < arg->desc->mt) )
                    return start+p;
                else 
                    return arg->desc->mt;
            } 
            
            /* If it is the 'local' diagonal block, we go to 1 */

        case 1:
            if ( (ls == 1) || (start == pivot) ) {
                tmp = arg->llvl->prevpiv(arg->llvl, lpivot / a, k, lstart / a);
                if ( tmp != arg->llvl->ldd ) 
                    return tmp * a * p + rpivot;
            }             
            
            if (ls != 0)
                startts = pivot;
            
            /* Search for predecessor in TS tree */
            if ( ( startts+p < arg->desc->mt ) && ( (((startts+p) / p) % a) != 0 ) )
                return startts + p;
            
        default:
            return arg->desc->mt;
        }
};

/****************************************************
 ***************************************************/
#define ENDCHECK( test, ret )                   \
    if ( !test )                                \
        exit( -1 );


int dplasma_qr_check( tiled_matrix_desc_t *A, qr_piv_t *qrpiv)
{
    int minMN = min(A->mt, A->nt );
    int i, m, k, nb;
    int check;

    int a = qrpiv->a;
    int p = qrpiv->p;

    /* 
     * Check Formula for NB geqrt 
     */
    fflush(stdout);
    {
        /*dplasma_qr_print_type( A, qrpiv );*/
        check = 1;
        for (k=0; k<minMN; k++) {
            nb = 0;
            for (m=k; m < A->mt; m++) {
                if ( dplasma_qr_gettype(qrpiv->a, qrpiv->p, k, m) > 0 )
                    nb++;
            }

            if ( nb != dplasma_qr_getnbgeqrf( a, p, k, A->mt) ) {
                check = 0;
                printf(" ----------------------------------------------------\n"
                       "  - a = %d, p = %d, M = %d, N = %d\n"
                       "     Check number of geqrt:\n"
                       "       For k=%d => return %d instead of %d",
                       a, p, A->mt, A->nt, k, dplasma_qr_getnbgeqrf( a, p, k, A->mt), nb );
            }
        }
        
        ENDCHECK( check, 1 );
    }

    /* 
     * Check indices of geqrt 
     */
    {
        check = 1;
        for (k=0; k<minMN; k++) {
            /* dplasma_qr_print_geqrt_k( A, qrpiv, k ); */
            nb = dplasma_qr_getnbgeqrf( a, p, k, A->mt );
            for (i=0; i < nb; i++) {

                m = dplasma_qr_getm( a, p, k, i );
                if ( m < k ) {
                    check = 0;
                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Check indices of geqrt:\n"
                           "        getm( k=%d, i=%d ) => m = %d", 
                           a, p, A->mt, A->nt, k, i, m);
                } else if ( i != dplasma_qr_geti( a, p, k, m) ) {
                    check = 0;
                    printf(" ----------------------------------------------------\n"
                           "  - a = %d, p = %d, M = %d, N = %d\n"
                           "     Check indices of geqrt:\n"
                           "        getm( k=%d, i=%d ) => m = %d && geti( k=%d, m=%d ) => i = %d\n", 
                           a, p, A->mt, A->nt, 
                           k, i, m, k, m, dplasma_qr_geti( a, p, k, m));
                }
            }
        }
        ENDCHECK( check, 2 );
    }

    /* 
     * Check next/prev
     */
    fflush(stdout);
    {
        int start, next, prev;
        check = 1;

        for (k=0; k<minMN; k++) {
            /* dplasma_qr_print_next_k( A, qrpiv, k); */
            /* dplasma_qr_print_prev_k( A, qrpiv, k); */

            start = A->mt;
            for(m=k; m<A->mt; m++) {

                do {
                    next = qrpiv->nextpiv(qrpiv, m, k, start);
                    if ( next == A->mt ) 
                        prev = qrpiv->prevpiv(qrpiv, m, k, m);
                    else 
                        prev = qrpiv->prevpiv(qrpiv, m, k, next);
                    
                    if ( start != prev ) {
                        printf(" ----------------------------------------------------\n"
                               "  - a = %d, p = %d, M = %d, N = %d\n"
                               "     Check next/prev:\n"
                               "       next( m=%d, k=%d, start=%d ) => %d && prev( m=%d, k=%d, start=%d ) => %d\n ( %d != %d )", 
                               a, p, A->mt, A->nt, 
                               m, k, start, next, m, k, next, prev, start, prev);
                        check = 0;
                    }
                    start = next;
                } while ( start != A->mt );
            }
        }
        ENDCHECK( check, 3 );
    }

    return 0;
}


qr_piv_t *dplasma_pivgen_init( tiled_matrix_desc_t *A, int type_llvl, int type_hlvl, int a, int p )
{
    qr_piv_t *qrpiv = (qr_piv_t*) malloc( sizeof(qr_piv_t) );
    qrpiv->currpiv = dplasma_qr_currpiv;
    qrpiv->nextpiv = dplasma_qr_nextpiv;
    qrpiv->prevpiv = dplasma_qr_prevpiv;
    qrpiv->desc = A;

    a = max( a, 1 );
    p = max( p, 1 );
    
    qrpiv->a = a;
    qrpiv->p = p;
    (void)type_hlvl;

    qrpiv->llvl = (qr_subpiv_t*) malloc( sizeof(qr_subpiv_t) );
    qrpiv->hlvl = NULL;

    if ( p > 1 ) {
      fprintf(stderr, "p > 1 not supported yet\n");
      exit(-1);
    }

    int local_mt = (A->mt + p * a - 1) / ( p * a );
    /*printf("Super tiles : %d\n", local_mt);*/

    switch( type_llvl ) {
    case DPLASMA_GREEDY_TREE :
        printf("Low level: Greedy\n");
        dplasma_greedy_init(qrpiv->llvl, local_mt, min(A->mt, A->nt), a);
        break;
    case DPLASMA_FIBONACCI_TREE :
        printf("Low level: Fibonacci\n");
        dplasma_fibonacci_init(qrpiv->llvl, local_mt, min(A->mt, A->nt), a);
        break;
    case DPLASMA_BINARY_TREE :
        printf("Low level: Binary\n");
        dplasma_binary_init(qrpiv->llvl, local_mt, a);
        break;
    case DPLASMA_FLAT_TREE :
    default:
        printf("Low level: Flat\n");
        dplasma_flat_init(qrpiv->llvl, local_mt, a);
    }

    if ( dplasma_qr_check( A, qrpiv ) != 0 )
        exit(-1);

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
    printf("\n------------ Localization = Type of pivot --------------\n");
    for(m=0; m<A->mt; m++) {
        printf("%4d | ", m);              
        for (k=0; k<min(minMN, m+1); k++) {
            printf( "%3d ", dplasma_qr_gettype( qrpiv->a, qrpiv->p, k, m ) );
        }
        for (k=min(minMN, m+1); k<minMN; k++) {
            printf( "    " );
        }
      
        printf("    ");
        printf("%2d,%4d | ", rank, lmg);
        for (k=0; k<min(minMN, lmg+1); k++) {
            printf( "%3d ", dplasma_qr_gettype( qrpiv->a, qrpiv->p, k, lmg) );
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
            printf( "%3d ", qrpiv->currpiv(qrpiv, m, k) );
        }
        for (k=min(minMN, m+1); k<minMN; k++) {
            printf( "    " );
        }
            
        printf("    ");
        printf("%2d,%4d | ", rank, lmg);
        for (k=0; k<min(minMN, lmg+1); k++) {
            printf( "%3d ", qrpiv->currpiv(qrpiv, lmg, k) );
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
            printf( "%4d  ", qrpiv->nextpiv(qrpiv, m, k, s) );
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
            printf( "%4d  ", qrpiv->prevpiv(qrpiv, m, k, s) );
        }
        printf("\n");
    }
}


void dplasma_qr_print_nbgeqrt( tiled_matrix_desc_t *A, qr_piv_t *qrpiv )
{
    int minMN = min(A->mt, A->nt );
    int m, k, nb;
    for (k=0; k<minMN; k++) {
        printf( "%3d ", k );
    }
    printf( "\n" );
    for (k=0; k<minMN; k++) {
        nb = 0;
        for (m=k; m < A->mt; m++) {
            if ( dplasma_qr_gettype(qrpiv->a, qrpiv->p, k, m) > 0 )
                nb++;
        }
        printf( "%3d ", nb );
    }
    printf( "\n" );
    for (k=0; k<minMN; k++) {
        printf( "%3d ", dplasma_qr_getnbgeqrf( qrpiv->a, qrpiv->p, k, A->mt) );
    }
    printf( "\n" );
}

void dplasma_qr_print_geqrt_k( tiled_matrix_desc_t *A, qr_piv_t *qrpiv, int k )
{
    int i, m, nb;

    printf( "k=%3d: ", k );
    
    printf( "  m:");
    nb = dplasma_qr_getnbgeqrf( qrpiv->a, qrpiv->p, k, A->mt );
    for (i=0; i < nb; i++) {
        m = dplasma_qr_getm( qrpiv->a, qrpiv->p, k, i );
        if ( i == dplasma_qr_geti( qrpiv->a, qrpiv->p, k, m) )
            printf( "%3d ", m );
        else
            printf( "x%2d ", dplasma_qr_geti( qrpiv->a, qrpiv->p, k, m) );
    }
    printf( "\n" );
}


