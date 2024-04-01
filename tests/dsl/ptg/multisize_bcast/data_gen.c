/*
 * Copyright (c) 2018-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "data_gen.h"
#include "stdarg.h"

#include <assert.h>
#include <stdlib.h>

/*
 *  mb, whole matrix row/column number, mt, each tile row/column number
 */
parsec_matrix_block_cyclic_t *create_and_distribute_data(int rank, int world, int mb, int mt)
{
    parsec_matrix_block_cyclic_t *m = (parsec_matrix_block_cyclic_t*)malloc(sizeof(parsec_matrix_block_cyclic_t));

    parsec_matrix_block_cyclic_init(m, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                                    rank,
                                    mb, mb,
                                    mt*mb, mt*mb,
                                    0, 0,
                                    mt*mb, mt*mb,
                                    1, world,
                                    1, 1, 0, 0);

    m->mat = parsec_data_allocate((size_t)m->super.nb_local_tiles *
                                  (size_t)m->super.bsiz *
                                  (size_t)parsec_datadist_getsizeoftype(m->super.mtype));

    return (parsec_matrix_block_cyclic_t*)m;
}

void free_data(parsec_matrix_block_cyclic_t *d)
{
    parsec_data_collection_destroy(&d->super.super);
    free(d);
}
