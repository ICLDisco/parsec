/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "a2a_data.h"
#include <stdarg.h>
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#include <assert.h>

parsec_tiled_matrix_dc_t *create_and_distribute_data(int rank, int world, int size)
{
    two_dim_block_cyclic_t *m = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));

    two_dim_block_cyclic_init(m, matrix_ComplexDouble, matrix_Tile,
                              world, rank,
                              size, 1, size, 1, rank, 1, world*size, 1,
                              1, 1, world);
    return (parsec_tiled_matrix_dc_t*)m;
}

void free_data(parsec_tiled_matrix_dc_t *d)
{
    parsec_data_collection_destroy(&d->super);
    free(d);
}
