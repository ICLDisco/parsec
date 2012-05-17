/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
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
    int ib = T->mb;
    /* 
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf 
     */

    object = dague_zgelqf_new( *A, (dague_ddesc_t*)A, *T, (dague_ddesc_t*)T, 
                               ib, NULL, NULL);

    object->p_tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_tau, T->nb * sizeof(Dague_Complex64_t) );

    object->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_work, ib * T->nb * sizeof(Dague_Complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zgelqf_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    /* Lower triangular part of tile with diagonal */
    dplasma_add2arena_lower( object->arenas[DAGUE_zgelqf_LOWER_TILE_ARENA], 
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* Upper triangular part of tile without diagonal */
    dplasma_add2arena_upper( object->arenas[DAGUE_zgelqf_UPPER_TILE_ARENA], 
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

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
    dplasma_progress(dague);

    dplasma_zgelqf_Destruct( dague_zgelqf );
    return 0;
}

void
dplasma_zgelqf_Destruct( dague_object_t *o )
{
    dague_zgelqf_object_t *dague_zgelqf = (dague_zgelqf_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgelqf->arenas[DAGUE_zgelqf_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgelqf->arenas[DAGUE_zgelqf_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgelqf->arenas[DAGUE_zgelqf_UPPER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgelqf->arenas[DAGUE_zgelqf_LITTLE_T_ARENA  ]->opaque_dtt) );
      
    dague_private_memory_fini( dague_zgelqf->p_work );
    dague_private_memory_fini( dague_zgelqf->p_tau  );
    free( dague_zgelqf->p_work );
    free( dague_zgelqf->p_tau  );
 
    DAGUE_INTERNAL_OBJECT_DESTRUCT(dague_zgelqf);
}

