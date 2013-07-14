/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "zgetrf_nopiv.h"

dague_object_t* dplasma_zgetrf_nopiv_New(tiled_matrix_desc_t *A,
                                         int *INFO)
{
    dague_zgetrf_nopiv_object_t *dague_getrf_nopiv;

    dague_getrf_nopiv = dague_zgetrf_nopiv_new( (dague_ddesc_t*)A,
                                                32, INFO );

    /* A */
    dplasma_add2arena_tile( dague_getrf_nopiv->arenas[DAGUE_zgetrf_nopiv_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    return (dague_object_t*)dague_getrf_nopiv;
}

void
dplasma_zgetrf_nopiv_Destruct( dague_object_t *o )
{
    dague_zgetrf_nopiv_object_t *dague_zgetrf_nopiv = (dague_zgetrf_nopiv_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgetrf_nopiv->arenas[DAGUE_zgetrf_nopiv_DEFAULT_ARENA]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(dague_zgetrf_nopiv);
}

int dplasma_zgetrf_nopiv( dague_context_t *dague,
                          tiled_matrix_desc_t *A )
{
    dague_object_t *dague_zgetrf_nopiv = NULL;

    int info = 0;
    dague_zgetrf_nopiv = dplasma_zgetrf_nopiv_New(A, &info);

    dague_enqueue( dague, (dague_object_t*)dague_zgetrf_nopiv);
    dplasma_progress(dague);

    return info;
}
