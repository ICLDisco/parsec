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

#define PIV(_ipiv, _mt, _i, _k) (_piv)[(_k)*(_mt)+(_i)]

/****************************************************
 *             Common ipiv
 ***************************************************/
/* Return the pivot to use for the row m at step k */
inline static int currpiv( const qr_piv_t *qrpiv, const int m, const int k ) {
    return (qrpiv->ipiv)[ k * (qrpiv->desc->mt) + m ];
}

/* Return the last row which has used the row m as a pivot in step k before the row start */
inline static int prevpiv( const qr_piv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    for( i=start+1; i<(qrpiv->desc->mt); i++ )
        if ( (qrpiv->ipiv)[i +  k * (qrpiv->desc->mt)] == p )
            return i;
    return i;
 }

/* Return the next row which will use the row m as a pivot in step k after it has been used by row start */
inline static int nextpiv( const qr_piv_t *qrpiv, const int p, const int k, const int start ) {
    int i;
    for( i=start-1; i>k; i-- )
        if ( (qrpiv->ipiv)[i + k* (qrpiv->desc->mt)] == p )
            return i;
    return (qrpiv->desc->mt);
}

/****************************************************
 *                 DPLASMA_FLAT_TREE
 ***************************************************/
int dplasma_flat_currpiv(const qr_piv_t *arg, const int m, const int k) 
{ 
    (void)arg;
    (void)m;
    return k;
};

int dplasma_flat_nextpiv(const qr_piv_t *arg, const int p, const int k, const int start)
{ 
    if ( ( p == k ) && (start > k+1 ) )
        return start-1;
    else
        return arg->desc->mt;
};

int dplasma_flat_prevpiv(const qr_piv_t *arg, const int p, const int k, const int start)
{ 
    if ( p == k ) 
      return start+1;
    else 
      return arg->desc->mt;
};

void dplasma_flat_init(qr_piv_t *arg, tiled_matrix_desc_t *descA){
    arg->currpiv = dplasma_flat_currpiv;
    arg->nextpiv = dplasma_flat_nextpiv;
    arg->prevpiv = dplasma_flat_prevpiv;
    arg->desc = descA;
    arg->ipiv = NULL;
};

/****************************************************
 *                 DPLASMA_BINARY_TREE
 ***************************************************/
int dplasma_binary_currpiv(const qr_piv_t *arg, const int m, const int k) 
{ 
    int tmp1 = m-k;
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

int dplasma_binary_nextpiv(const qr_piv_t *arg, const int p, const int k, const int start)
{ 
    int step, tmpp;

#ifndef PRINT_PIVGEN
    assert( (start == arg->desc->mt) || (dplasma_binary_currpiv( arg, start, k ) == p) );
#else
    /* Otherwise, it has just been used as a pivot so the interval should be a power of 2 */
    if ( start != arg->desc->mt && dplasma_binary_currpiv( arg, start, k ) != p ) 
        return -1;
#endif

    if ( start <= p )
        return arg->desc->mt;

    if (start == arg->desc->mt) {
        step = 1;
        tmpp = p - k;
    } else {
        step = ( start - p ) << 1;
        tmpp = ( p     - k ) >> 1;
    }

    /* Case of the first line */
    if ( tmpp == 0 )  {
        if ( p+step <  arg->desc->mt )
            return p + step;
        else 
            return arg->desc->mt;
    }

    while( (tmpp % 2 == 0) && ( p + step >= arg->desc->mt ) ) {
        step = step << 1;
        tmpp = tmpp >> 1;
    }

    if ( ( tmpp%2 != 0 ) || ( (p - step) == k ) ) 
        return  arg->desc->mt;
    else
        return p + step;
};

int dplasma_binary_prevpiv(const qr_piv_t *arg, const int p, const int k, const int start)
{ 
#ifndef PRINT_PIVGEN
    assert( start >= p && ( start == p || dplasma_binary_currpiv( arg, start, k ) == p));
#else
    /* Otherwise, it has just been used as a pivot so the interval should be a power of 2 */
    if ( start < p || ( (start > p) && (dplasma_binary_currpiv( arg, start, k ) != p) ) )
        return -1;
#endif

    if ( (start == p) && ( (p-k)%2 == 0 ) ) {
        int i, bit, tmp;
        if ((p-k) == 0)
            bit = (int)( log( (double)(arg->desc->mt - k) ) / log( 2. ) );
        else 
            bit = (int)( log( (double)(p-k) ) / log( 2. ) );

        for( i=bit; i>-1; i--){
            tmp = (p-k) | (1 << i);
            if ( ( (p-k) != tmp ) && ( tmp+k < arg->desc->mt ) )
                return tmp+k;
        }                
        return arg->desc->mt;
    }


    if ( (start - p) > 1 )
        return p + ( (start-p) >> 1 );
    else {
        return arg->desc->mt;
    }
};

void dplasma_binary_init(qr_piv_t *arg, tiled_matrix_desc_t *descA){
    arg->currpiv = dplasma_binary_currpiv;
    arg->nextpiv = dplasma_binary_nextpiv;
    arg->prevpiv = dplasma_binary_prevpiv;
    arg->desc = descA;
    arg->ipiv = NULL;
};

/****************************************************
 *          DPLASMA_FIBONACCI_TREE
 ***************************************************/
void dplasma_fibonacci_init(qr_piv_t *arg, tiled_matrix_desc_t *A){
    int minMN = min(A->mt, A->nt);
    int *ipiv;

    arg->currpiv = currpiv;
    arg->nextpiv = nextpiv;
    arg->prevpiv = prevpiv;
    arg->desc = A;
    arg->ipiv = (int*)malloc( A->mt * minMN * sizeof(int) );
    ipiv = arg->ipiv;
    memset(ipiv, 0, A->mt*minMN*sizeof(int));
   
    {
        int f0, f1, f2, k, m;
        
        /* Fill in the first column */
        f0 = 0;
        f1 = 1;
        for (m=1; m < A->mt; ) {
            for (k=0; (k < f1) && (m < A->mt); k++, m++) {
                ipiv[m] = m - f1;
            }
            f2 = f0 + f1;
            f0 = f1;
            f1 = f2;
        }

        for( k=1; k<minMN; k++) {
            for(m=k+1; m < A->mt; m++) {
                ipiv[ k * A->mt + m ] = ipiv[ (k-1) * A->mt + m - 1 ] + 1;
            }
        }
    }
};

/****************************************************
 *          DPLASMA_GREEDY_TREE
 ***************************************************/
void dplasma_greedy_init(qr_piv_t *arg, tiled_matrix_desc_t *A){
    int minMN = min(A->mt, A->nt);
    int *ipiv;

    arg->currpiv = currpiv;
    arg->nextpiv = nextpiv;
    arg->prevpiv = prevpiv;
    arg->desc = A;
    arg->ipiv = (int*)malloc( A->mt * minMN * sizeof(int) );
    ipiv = arg->ipiv;
    memset(ipiv, 0, A->mt*minMN*sizeof(int));
   
    {
        int j, k, height, start, end, firstk = 0;
        int *nT = (int*)malloc(minMN*sizeof(int));
        int *nZ = (int*)malloc(minMN*sizeof(int));
        memset( nT, 0, minMN*sizeof(int));
        memset( nZ, 0, minMN*sizeof(int));
        nT[0] = A->mt;
        
        k = 0;
        while ( ! ( ( nT[minMN-1] == A->mt-minMN+1 ) &&
                    ( nZ[minMN-1]+1 == nT[minMN-1] ) ) ) {
            height = (nT[k] - nZ[k]) / 2;
            if ( height == 0 ) {
                while ( (firstk < minMN) &&
                        ( nT[firstk] == A->mt-firstk ) && 
                        ( nZ[firstk]+1 == nT[firstk] ) ) {
                    firstk++;
                }
                k = firstk;
                continue;
            }
             
            if (k < minMN-1) nT[k+1] += height;
            start = A->mt - 1 - nZ[k];
            end = start -height;
            nZ[k] += height;
            
            for( j=start; j > end; j-- )
                ipiv[ k*A->mt + j ] = j - height;

            k++;
            if (k > minMN-1) k = firstk;
        }
        
        free(nT);
        free(nZ);
    }
};

/****************************************************
 ***************************************************/
qr_piv_t *dplasma_pivgen_init( int type_llvl, int type_hlvl, tiled_matrix_desc_t *A )
{
    qr_piv_t *qrpiv = (qr_piv_t*) malloc( sizeof(qr_piv_t) );
    (void)type_hlvl;
    
    switch( type_llvl ) {
    case DPLASMA_GREEDY_TREE :
        printf("Low level: Greedy\n");
        dplasma_greedy_init( qrpiv, A );
        break;
    case DPLASMA_FIBONACCI_TREE :
        printf("Low level: Fibonacci\n");
        dplasma_fibonacci_init( qrpiv, A );
        break;
    case DPLASMA_BINARY_TREE :
        printf("Low level: Binary\n");
        dplasma_binary_init( qrpiv, A );
        break;
    case DPLASMA_FLAT_TREE :
    default:
        printf("Low level: Flat\n");
        dplasma_flat_init( qrpiv, A );
    }

#ifdef PRINT_PIVGEN
    {
        int minMN = min(A->mt, A->nt );
        int m, k;
        for(m=0; m<A->mt; m++) {
            printf("%4d | ", m);              
            for (k=0; k<min(minMN, m+1); k++) {
                printf( "%4d  ", qrpiv->currpiv(qrpiv, m, k) );
            }
            printf("\n");
        }
    }

    printf("\n------------ Next --------------\n");
    {
        int m, k;
        printf( "       " );
        for(k=A->mt; k>0; k--)
            printf( "%4d  ", k );
        printf( "\n" );
        for(m=0; m<A->mt; m++) {
            printf("%4d | ", m);              
            for(k=A->mt; k>0; k--) {
                /*for (k=0; k<min(minMN, m+1); k++) {*/
                printf( "%4d  ", qrpiv->nextpiv(qrpiv, m, 0, k) );
            }
            printf("\n");
        }
    }

    printf("\n------------ prev --------------\n");
    {
        int m, k;
        printf( "       " );
        for(k=A->mt; k>-1; k--)
            printf( "%4d  ", k );
        printf( "\n" );

        for(m=0; m<A->mt; m++) {
            printf("%4d | ", m);              
            for(k=A->mt; k>-1; k--) {
                /*for (k=0; k<min(minMN, m+1); k++) {*/
                printf( "%4d  ", qrpiv->prevpiv(qrpiv, m, 0, k) );
            }
            printf("\n");
        }
    }
#endif

    return qrpiv;
}

void dplasma_pivgen_finalize( qr_piv_t *qrpiv )
{
    if ( qrpiv->ipiv != NULL )
        free( qrpiv->ipiv );
}
