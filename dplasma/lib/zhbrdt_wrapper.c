/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */


#include <core_blas.h>
#include "data_dist/matrix/matrix.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dague/private_mempool.h"

#include "zhbrdt.h"

dague_handle_t* dplasma_zhbrdt_New(tiled_matrix_desc_t* A /* data A */)
{
    dague_zhbrdt_handle_t *dague_zhbrdt = NULL;

    dague_zhbrdt = dague_zhbrdt_new(A, A->mb-1);

    dplasma_add2arena_rectangle( dague_zhbrdt->arenas[DAGUE_zhbrdt_DEFAULT_ARENA],
                                 (A->nb)*(A->mb)*sizeof(dague_complex64_t), 16,
                                 dague_datatype_double_complex_t,
                                 A->mb, A->nb, -1 );
    return (dague_handle_t*)dague_zhbrdt;
}

void dplasma_zhbrdt_Destruct( dague_handle_t *handle )
{
    dague_handle_free(handle);
}

