/*
 * Copyright (c) 2011      The University of Tennessee and The University
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

#include "zgelqf.h"

dague_object_t* dplasma_zgelqf_New( tiled_matrix_desc_t *A,
                                    tiled_matrix_desc_t *T )
{
    dague_zgelqf_object_t* object;
    
    /* 
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf 
     */

    object = dague_zgelqf_new( (dague_ddesc_t*)A, (dague_ddesc_t*)T, 
                               A->mt, A->nt, A->mb, A->nb, A->m, A->n, 
                               T->mb, T->nb, T->mb /*ib*/, NULL, NULL);

    object->pool_Tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->pool_Tau, T->nb * sizeof(Dague_Complex64_t) );

    object->pool_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->pool_work, T->mb * T->nb * sizeof(Dague_Complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zgelqf_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( object->arenas[DAGUE_zgelqf_LOWER_TILE_ARENA], 
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( object->arenas[DAGUE_zgelqf_UPPER_TILE_ARENA], 
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgelqf_LITTLE_T_ARENA], 
                                 T->mb*T->nb*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, T->mb, T->nb, -1);

    return (dague_object_t*)object;
}

int dplasma_zgelqf( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *T) 
{
    dague_object_t *dague_zgelqf = NULL;

    dague_zgelqf = dplasma_zgelqf_New(A, T);

    dague_enqueue(dague, (dague_object_t*)dague_zgelqf);
    dague_progress(dague);

    dplasma_zgelqf_Destruct( dague_zgelqf );
    return 0;
}

void
dplasma_zgelqf_Destruct( dague_object_t *o )
{
    dague_zgelqf_object_t *dague_zgelqf = (dague_zgelqf_object_t *)o;

    dague_private_memory_fini( dague_zgelqf->pool_work );
    dague_private_memory_fini( dague_zgelqf->pool_Tau  );
    free( dague_zgelqf->pool_work );
    free( dague_zgelqf->pool_Tau  );

    dague_zgelqf_destroy(dague_zgelqf);
}

