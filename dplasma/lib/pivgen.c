/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include <plasma.h>
#include <dague.h>
#include "dplasma.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"

#ifndef min
#define min(__a, __b) ( ( (__a) < (__b) ) ? (__a) : (__b) )
#endif

void dplasma_pivgen( int type, tiled_matrix_desc_t *A, int *ipiv )
{
    int minMN = min(A->mt, A->nt);
    memset(ipiv, 0, A->mt*minMN*sizeof(int));

    switch( type ) {
    case DPLASMA_GREEDY_TREE :
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
             
            nT[k+1] += height;
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
    break;
    case DPLASMA_FIBONACCI_TREE :
    case DPLASMA_BINARY_TREE :
    case DPLASMA_FLAT_TREE :
    default:
    {
        int m, n;
        int *ipiv2 = ipiv;
        for(n=0; n<minMN; n++) {
            for (m=0; m<A->mt; m++) {
                *ipiv2 = n; ipiv2++;
            }
        }
    }
    }


#ifdef 0
    {
        int m, k;
        for(m=0; m<A->mt; m++) {
            printf("%4d | ", m);              
            for (k=0; k<min(minMN, m+1); k++) {
                printf( "%4d  ", ipiv[k*A->mt + m]);
            }
            printf("\n");
        }
    }
#endif

    return;
}
