/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <plasma.h>
#include <dague.h>
#include <scheduling.h>
#include "data_dist/matrix/matrix.h"
#include "dplasma_z.h"
#include "dplasmatypes.h"
#include "dplasmaaux.h"
#include "memory_pool.h"

#include "generated/zhbrdt.h"

dague_object_t* dplasma_zhbrdt_New(tiled_matrix_desc_t* A /* data A */)
{
    dague_zhbrdt_object_t *dague_zhbrdt = NULL;

    dague_zhbrdt = dague_zhbrdt_new(A);

    dplasma_add2arena_tile(dague_zhbrdt->arenas[DAGUE_zhbrdt_DEFAULT_ARENA],
                           (A->nb)*(A->nb)*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->nb);

    return (dague_object_t*)dague_zhbrdt;
}

void dplasma_zhbrdt_Destruct( dague_object_t* o )
{
    dague_zhbrdt_destroy( (dague_zhbrdt_object_t*)o );
}

