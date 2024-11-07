/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "a2a_data.h"
#include <stdarg.h>
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#include <assert.h>

parsec_tiled_matrix_t *create_and_distribute_data(int rank, int world, int size)
{
    parsec_matrix_block_cyclic_t *m = (parsec_matrix_block_cyclic_t*)malloc(sizeof(parsec_matrix_block_cyclic_t));

    parsec_matrix_block_cyclic_init(m, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                              rank,
                              size, 1, size, 1, rank, 1, world*size, 1,
                              world, 1, 1, 1, 0, 0);
    return (parsec_tiled_matrix_t*)m;
}

void free_data(parsec_tiled_matrix_t *d)
{
    parsec_data_collection_destroy(&d->super);
    free(d);
}
