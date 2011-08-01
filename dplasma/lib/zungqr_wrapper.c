/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include "zungqr.h"

dague_object_t* dplasma_zungqr_New( tiled_matrix_desc_t *A,
                                    tiled_matrix_desc_t *T,
                                    tiled_matrix_desc_t *Q)
{
    dague_zungqr_object_t* object;
    int ib = T->mb;
    /* 
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf 
     */

    object = dague_zungqr_new( *A, (dague_ddesc_t*)A, 
                               *T, (dague_ddesc_t*)T, 
                               *Q, (dague_ddesc_t*)Q, 
                               ib, NULL);

    object->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_work, ib * T->nb * sizeof(Dague_Complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zungqr_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( object->arenas[DAGUE_zungqr_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* /\* Upper triangular part of tile with diagonal *\/ */
    /* dplasma_add2arena_upper( object->arenas[DAGUE_zungqr_UPPER_TILE_ARENA],  */
    /*                          A->mb*A->nb*sizeof(Dague_Complex64_t), */
    /*                          DAGUE_ARENA_ALIGNMENT_SSE, */
    /*                          MPI_DOUBLE_COMPLEX, A->mb, 1 ); */

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zungqr_LITTLE_T_ARENA], 
                                 T->mb*T->nb*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, T->mb, T->nb, -1);

    return (dague_object_t*)object;
}

int dplasma_zungqr( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T, tiled_matrix_desc_t *Q) 
{
    dague_object_t *dague_zungqr = NULL;

    dague_zungqr = dplasma_zungqr_New(A, T, Q);

    dague_enqueue(dague, (dague_object_t*)dague_zungqr);
    dplasma_progress(dague);

    dplasma_zungqr_Destruct( dague_zungqr );
    return 0;
}

void
dplasma_zungqr_Destruct( dague_object_t *o )
{
    dague_zungqr_object_t *dague_zungqr = (dague_zungqr_object_t *)o;

    dague_private_memory_fini( dague_zungqr->p_work );
    free( dague_zungqr->p_work );
    dague_zungqr_destroy(dague_zungqr);
}

