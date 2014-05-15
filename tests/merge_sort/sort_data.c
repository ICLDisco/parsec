/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "sort_data.h"
#include "stdarg.h"
#include <data_dist/matrix/two_dim_rectangle_cyclic.h>

#include <assert.h>
#include <stdlib.h>

tiled_matrix_desc_t *create_and_distribute_data(int rank, int world, int mb, int mt, int typesize)
{
    two_dim_block_cyclic_t *m = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    
    two_dim_block_cyclic_init(m, matrix_ComplexDouble, matrix_Tile,
                              world, rank,
                              mb*typesize, 1,
                              mt*mb*typesize, 1,
                              0, 0,
                              mt*mb*typesize, 1,
                              1, 1,
                              world);

    m->mat = dague_data_allocate((size_t)m->super.nb_local_tiles *
                                (size_t)m->super.bsiz *
                                (size_t)dague_datadist_getsizeoftype(m->super.mtype));

    return (tiled_matrix_desc_t*)m;
}

void free_data(tiled_matrix_desc_t *d)
{
    dague_ddesc_destroy(&d->super);
    free(d);
}
