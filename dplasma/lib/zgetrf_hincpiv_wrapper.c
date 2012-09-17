/*
 * Copyright (c) 2012      The University of Tennessee and The University
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

#include "zgetrf_hincpiv.h"

dague_object_t* dplasma_zgetrf_hincpiv_New( qr_piv_t *qrpiv,
                                            tiled_matrix_desc_t *A,
                                            tiled_matrix_desc_t *IPIV,
                                            tiled_matrix_desc_t *LT,
                                            int* INFO )
{
    dague_zgetrf_hincpiv_object_t* object;
    int ib = LT->mb;

    /*
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf
     */

    object = dague_zgetrf_hincpiv_new( *A,  (dague_ddesc_t*)A,
                                            (dague_ddesc_t*)IPIV,
                                       *LT, (dague_ddesc_t*)LT,
                                       qrpiv, ib,
                                       NULL, NULL,
                                       INFO);

    object->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_work, ib * LT->nb * sizeof(dague_complex64_t) );

    object->p_tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_tau, LT->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zgetrf_hincpiv_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( object->arenas[DAGUE_zgetrf_hincpiv_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( object->arenas[DAGUE_zgetrf_hincpiv_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgetrf_hincpiv_SMALL_L_ARENA],
                                 LT->mb*LT->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, LT->mb, LT->nb, -1);

    /* IPIV */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgetrf_hincpiv_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    return (dague_object_t*)object;
}

int dplasma_zgetrf_hincpiv( dague_context_t *dague,
                            qr_piv_t *qrpiv,
                            tiled_matrix_desc_t *A,
                            tiled_matrix_desc_t *IPIV,
                            tiled_matrix_desc_t *LT,
                            int* INFO )
{
    dague_object_t *dague_zgetrf_hincpiv = NULL;

    dague_zgetrf_hincpiv = dplasma_zgetrf_hincpiv_New(qrpiv, A, IPIV, LT, INFO);

    dague_enqueue(dague, (dague_object_t*)dague_zgetrf_hincpiv);
    dplasma_progress(dague);

    dplasma_zgetrf_hincpiv_Destruct( dague_zgetrf_hincpiv );
    return 0;
}

void
dplasma_zgetrf_hincpiv_Destruct( dague_object_t *o )
{
    dague_zgetrf_hincpiv_object_t *dague_zgetrf_hincpiv = (dague_zgetrf_hincpiv_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgetrf_hincpiv->arenas[DAGUE_zgetrf_hincpiv_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_hincpiv->arenas[DAGUE_zgetrf_hincpiv_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_hincpiv->arenas[DAGUE_zgetrf_hincpiv_UPPER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_hincpiv->arenas[DAGUE_zgetrf_hincpiv_SMALL_L_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_hincpiv->arenas[DAGUE_zgetrf_hincpiv_PIVOT_ARENA     ]->opaque_dtt) );

    dague_private_memory_fini( dague_zgetrf_hincpiv->p_work );
    dague_private_memory_fini( dague_zgetrf_hincpiv->p_tau  );

    free( dague_zgetrf_hincpiv->p_work );
    free( dague_zgetrf_hincpiv->p_tau  );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

