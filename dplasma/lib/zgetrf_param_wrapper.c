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

#include "zgetrf_param.h"
#include "zgeqrf.h"

dague_object_t* dplasma_zgetrf_param_New( qr_piv_t *qrpiv,
                                          tiled_matrix_desc_t *A,
                                          tiled_matrix_desc_t *TS,
                                          tiled_matrix_desc_t *TT,
                                          dague_ddesc_t * IPIV, 
                                          int* INFO )
{
    dague_zgetrf_param_object_t* object;
    int ib = TS->mb;

    /* 
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf 
     */

    object = dague_zgetrf_param_new( *A,  (dague_ddesc_t*)A, 
                                     *TS, (dague_ddesc_t*)TS, 
                                     *TT, (dague_ddesc_t*)TT, 
                                     qrpiv, ib, NULL, NULL,
                                     IPIV, INFO, NULL);

    object->p_tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_tau, TS->nb * sizeof(Dague_Complex64_t) );

    object->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_work, ib * TS->nb * sizeof(Dague_Complex64_t) );

    object->work_pool = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->work_pool, TS->mb * TS->nb * sizeof(Dague_Complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zgetrf_param_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( object->arenas[DAGUE_zgetrf_param_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( object->arenas[DAGUE_zgetrf_param_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgetrf_param_SMALL_L_ARENA], 
                                 TS->mb*TS->nb*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, TS->mb, TS->nb, -1);

    return (dague_object_t*)object;
}

int dplasma_zgetrf_param( dague_context_t *dague, 
                          qr_piv_t *qrpiv,
                          tiled_matrix_desc_t *A, 
                          tiled_matrix_desc_t *TS,
                          tiled_matrix_desc_t *TT,
                          dague_ddesc_t * IPIV, 
                          int* INFO )
{
    dague_object_t *dague_zgetrf_param = NULL;

    dague_zgetrf_param = dplasma_zgetrf_param_New(qrpiv, A, TS, TT, IPIV, INFO);

    dague_enqueue(dague, (dague_object_t*)dague_zgetrf_param);
    dplasma_progress(dague);

    dplasma_zgetrf_param_Destruct( dague_zgetrf_param );
    return 0;
}

void
dplasma_zgetrf_param_Destruct( dague_object_t *o )
{
    dague_zgetrf_param_object_t *dague_zgetrf_param = (dague_zgetrf_param_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgetrf_param->arenas[DAGUE_zgetrf_param_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_param->arenas[DAGUE_zgetrf_param_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_param->arenas[DAGUE_zgetrf_param_UPPER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf_param->arenas[DAGUE_zgetrf_param_SMALL_L_ARENA  ]->opaque_dtt) );
      
    dague_private_memory_fini( dague_zgetrf_param->p_work );
    dague_private_memory_fini( dague_zgetrf_param->p_tau  );
    dague_private_memory_fini( dague_zgetrf_param->work_pool  );
    free( dague_zgetrf_param->p_work );
    free( dague_zgetrf_param->p_tau  );
    free( dague_zgetrf_param->work_pool );

    dague_zgetrf_param_destroy(dague_zgetrf_param);
}

