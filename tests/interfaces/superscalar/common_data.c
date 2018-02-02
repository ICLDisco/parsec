/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "parsec/parsec_config.h"
#include "common_data.h"
#include "stdarg.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

#include <assert.h>
#include <stdlib.h>

parsec_tiled_matrix_dc_t *create_and_distribute_data(int rank, int world, int mb, int mt)
{
    two_dim_block_cyclic_t *m = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    two_dim_block_cyclic_init(m, matrix_ComplexDouble, matrix_Tile,
                              world, rank,
                              mb, 1,
                              mt*mb, 1,
                              0, 0,
                              mt*mb, 1,
                              1, 1,
                              world);

    m->mat = parsec_data_allocate((size_t)m->super.nb_local_tiles *
                                (size_t)m->super.bsiz *
                                (size_t)parsec_datadist_getsizeoftype(m->super.mtype));

    return (parsec_tiled_matrix_dc_t*)m;
}

parsec_tiled_matrix_dc_t *create_and_distribute_empty_data(int rank, int world, int mb, int mt)
{
    two_dim_block_cyclic_t *m = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));
    two_dim_block_cyclic_init(m, matrix_ComplexDouble, matrix_Tile,
                              world, rank,
                              mb, 1,
                              mt*mb, 1,
                              0, 0,
                              mt*mb, 1,
                              1, 1,
                              world);

    return (parsec_tiled_matrix_dc_t*)m;
}

void free_data(parsec_tiled_matrix_dc_t *d)
{
    parsec_matrix_destroy_data(d);
    parsec_data_collection_destroy(&d->super);
    free(d);
}
