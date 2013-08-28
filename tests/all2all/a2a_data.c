/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "a2a_data.h"
#include "stdarg.h"
#include <data_dist/matrix/matrix.h>

#include <assert.h>

tiled_matrix_desc_t *create_and_distribute_data(int rank, int world, int size)
{
    tiled_matrix_desc_t *m = (tiled_matrix_desc_t*)malloc(sizeof(tiled_matrix_desc_t));
    
    tiled_matrix_desc_init(m, matrix_ComplexDouble, matrix_Tile,
                           tiled_matrix_desc_type, world, 1, rank,
                           size, 1, size, 1, rank, 1, world*size, 1);
    return m;
}

void free_data(tiled_matrix_desc_t *d)
{
    dague_ddesc_destroy(&d->super);
    free(d);
}
