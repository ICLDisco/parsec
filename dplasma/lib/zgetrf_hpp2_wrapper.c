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

#include "zgetrf_hpp2.h"

dague_object_t* dplasma_zgetrf_hpp2_New( qr_piv_t *qrpiv,
                                            tiled_matrix_desc_t *A,
                                            tiled_matrix_desc_t *IPIV,
                                            tiled_matrix_desc_t *LT,
                                            tiled_matrix_desc_t *LT2,
                                            int* INFO )
{
    dague_zgetrf_hpp2_object_t* object;
    int ib = LT->mb;

    /*
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf
     */

    object = dague_zgetrf_hpp2_new( *A,  (dague_ddesc_t*)A,
                                   (dague_ddesc_t*)IPIV,
                                   *LT, (dague_ddesc_t*)LT,
                                   *LT2, (dague_ddesc_t*)LT2,
                                   qrpiv, ib,
                                   NULL, NULL,
                                   INFO);

    object->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_work, ib * LT->nb * sizeof(Dague_Complex64_t) );

    object->p_tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_tau, LT->nb * sizeof(Dague_Complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zgetrf_hpp2_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( object->arenas[DAGUE_zgetrf_hpp2_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* IPIV */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgetrf_hpp2_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( object->arenas[DAGUE_zgetrf_hpp2_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgetrf_hpp2_LITTLE_T_ARENA],
                                 LT->mb*LT->nb*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, LT->mb, LT->nb, -1);

    return (dague_object_t*)object;
}

int dplasma_zgetrf_hpp2( dague_context_t *dague,
                            qr_piv_t *qrpiv,
                            tiled_matrix_desc_t *A,
                            tiled_matrix_desc_t *IPIV,
                            tiled_matrix_desc_t *LT,
                            tiled_matrix_desc_t *LT2,
                            int* INFO )
{
    dague_object_t *dague_zgetrf_hpp2 = NULL;

    dague_zgetrf_hpp2 = dplasma_zgetrf_hpp2_New(qrpiv, A, IPIV, LT, LT2, INFO);

    dague_enqueue(dague, (dague_object_t*)dague_zgetrf_hpp2);
    dplasma_progress(dague);

    dplasma_zgetrf_hpp2_Destruct( dague_zgetrf_hpp2 );
    return 0;
}

void
dplasma_zgetrf_hpp2_Destruct( dague_object_t *o )
{
    dague_zgetrf_hpp2_object_t *dague_zgetrf_hpp2 = (dague_zgetrf_hpp2_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgetrf_hpp2->arenas[DAGUE_zgetrf_hpp2_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_hpp2->arenas[DAGUE_zgetrf_hpp2_UPPER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_hpp2->arenas[DAGUE_zgetrf_hpp2_PIVOT_ARENA     ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_hpp2->arenas[DAGUE_zgetrf_hpp2_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_hpp2->arenas[DAGUE_zgetrf_hpp2_LITTLE_T_ARENA  ]->opaque_dtt) );

    dague_private_memory_fini( dague_zgetrf_hpp2->p_work );
    dague_private_memory_fini( dague_zgetrf_hpp2->p_tau  );

    free( dague_zgetrf_hpp2->p_work );
    free( dague_zgetrf_hpp2->p_tau  );

    dague_zgetrf_hpp2_destroy(dague_zgetrf_hpp2);
}

