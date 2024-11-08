/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 *
 */

#include "parsec/runtime.h"
#include "reduc_data.h"
#include "stdarg.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

parsec_tiled_matrix_t *create_and_distribute_data(int rank, int world, int mb, int mt, int typesize)
{
    parsec_matrix_block_cyclic_t *m = (parsec_matrix_block_cyclic_t*)malloc(sizeof(parsec_matrix_block_cyclic_t));

    parsec_matrix_block_cyclic_init(m, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                              rank,
                              mb*typesize, 1,
                              mt*mb*typesize, 1,
                              0, 0,
                              mt*mb*typesize, 1,
                              world, 1,
                              1, 1,
                              0, 0);

    m->mat = parsec_data_allocate((size_t)m->super.nb_local_tiles *
                                (size_t)m->super.bsiz *
                                (size_t)parsec_datadist_getsizeoftype(m->super.mtype));

    return (parsec_tiled_matrix_t*)m;
}

void free_data(parsec_tiled_matrix_t *d)
{
    parsec_data_collection_destroy(&d->super);
    free(d);
}
