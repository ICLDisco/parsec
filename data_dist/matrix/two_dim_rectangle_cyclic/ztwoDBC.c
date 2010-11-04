/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <stdint.h>
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

int twoDBC_ztolapack(two_dim_block_cyclic_t *Mdesc, Dague_Complex64_t* A, int lda)
{
    unsigned int i, j, il, jl, x, y;
    unsigned int imax, jmax;
    double *bdl, *f77;
    int64_t dec;

    /* check which tiles to generate */
    for ( j = 0 ; j < Mdesc->super.nt ; j++)
        for ( i = 0 ; i < Mdesc->super.mt ; i++)
        {
            if( Mdesc->super.super.myrank ==
                Mdesc->super.super.rank_of((dague_ddesc_t *)Mdesc, i, j ) )
            {
                il = i / ( Mdesc->nrst * Mdesc->GRIDrows ) +  (i % ( Mdesc->nrst * Mdesc->GRIDrows )) - ( Mdesc->nrst * Mdesc->rowRANK );
                jl = j / ( Mdesc->ncst * Mdesc->GRIDcols ) +  (j % ( Mdesc->ncst * Mdesc->GRIDcols )) - ( Mdesc->ncst * Mdesc->colRANK );
                dec = ((int64_t)(Mdesc->super.nb)*(int64_t)lda*(int64_t)(jl)) + (int64_t)((Mdesc->super.mb)*(il));
                bdl = Mdesc->super.super.data_of((dague_ddesc_t *)Mdesc, i, j );
                f77 = &A[ dec ];

                imax = ( i == Mdesc->super.mt-1 ) ? Mdesc->super.m - i * Mdesc->super.mb : Mdesc->super.mb ;
                jmax = ( j == Mdesc->super.nt-1 ) ? Mdesc->super.n - j * Mdesc->super.nb : Mdesc->super.nb ;
                for (y = 0; y < jmax; y++)
                  for (x = 0; x < imax; x++)
                    f77[lda*y+x] = bdl[(Mdesc->super.mb)*y + x];
            }
        }
    return 0;
}

