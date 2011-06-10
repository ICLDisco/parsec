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

    dague_zhbrdt = dague_zhbrdt_new(A, A->mb-1);

    dplasma_add2arena_rectangle( dague_zhbrdt->arenas[DAGUE_zhbrdt_DEFAULT_ARENA],
                                 (A->nb)*(A->mb)*sizeof(Dague_Complex64_t), 16,
                                 MPI_DOUBLE_COMPLEX, 
                                 A->mb, A->nb, -1 );
    return (dague_object_t*)dague_zhbrdt;
}

void dplasma_zhbrdt_Destruct( dague_object_t* o )
{
    dague_zhbrdt_destroy( (dague_zhbrdt_object_t*)o );
}

