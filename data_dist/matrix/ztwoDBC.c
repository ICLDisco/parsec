/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <stdint.h>
#include "parsec_config.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_internal.h"

int twoDBC_ztolapack(two_dim_block_cyclic_t *Mdesc, parsec_complex64_t* A, int lda)
{
    int i, j, il, jl, x, y;
    int imax, jmax;
    parsec_complex64_t *bdl, *f77;
    int64_t dec;
    parsec_data_t* data;

    /* check which tiles to generate */
    for ( j = 0 ; j < Mdesc->super.nt ; j++)
        for ( i = 0 ; i < Mdesc->super.mt ; i++) {
            if( Mdesc->super.super.myrank ==
                Mdesc->super.super.rank_of((parsec_ddesc_t *)Mdesc, i, j ) ) {
                il = i / ( Mdesc->grid.strows * Mdesc->grid.rows ) +  (i % ( Mdesc->grid.strows * Mdesc->grid.rows )) - ( Mdesc->grid.strows * Mdesc->grid.rrank );
                jl = j / ( Mdesc->grid.stcols * Mdesc->grid.cols ) +  (j % ( Mdesc->grid.stcols * Mdesc->grid.cols )) - ( Mdesc->grid.stcols * Mdesc->grid.crank );
                dec = ((int64_t)(Mdesc->super.nb)*(int64_t)lda*(int64_t)(jl)) + (int64_t)((Mdesc->super.mb)*(il));
                data = Mdesc->super.super.data_of((parsec_ddesc_t *)Mdesc, i, j );
                bdl = PARSEC_DATA_COPY_GET_PTR(data->device_copies[0]);
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

