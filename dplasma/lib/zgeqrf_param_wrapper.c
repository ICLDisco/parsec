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
#include "dplasma/lib/pivgen.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include "zgeqrf_param.h"

dague_object_t* dplasma_zgeqrf_param_New( int tree_llvl, int tree_hlvl,
                                          int sze_ts, int sze_hlvl,
                                          tiled_matrix_desc_t *A,
                                          tiled_matrix_desc_t *TS,
                                          tiled_matrix_desc_t *TT )
{
    dague_zgeqrf_param_object_t* object;
    qr_piv_t *qrpiv;
    int ib = TS->mb;

    qrpiv = dplasma_pivgen_init( A, tree_llvl, tree_hlvl, sze_ts, sze_hlvl );

    /* 
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf 
     */

    object = dague_zgeqrf_param_new( *A,  (dague_ddesc_t*)A, 
                                     *TS, (dague_ddesc_t*)TS, 
                                     *TT, (dague_ddesc_t*)TT, 
                                     qrpiv, ib, NULL, NULL);

    object->p_tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_tau, TS->nb * sizeof(Dague_Complex64_t) );

    object->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( object->p_work, ib * TS->nb * sizeof(Dague_Complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zgeqrf_param_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( object->arenas[DAGUE_zgeqrf_param_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( object->arenas[DAGUE_zgeqrf_param_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(Dague_Complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* Little T */
    dplasma_add2arena_rectangle( object->arenas[DAGUE_zgeqrf_param_LITTLE_T_ARENA], 
                                 TS->mb*TS->nb*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, TS->mb, TS->nb, -1);

    return (dague_object_t*)object;
}

int dplasma_zgeqrf_param( dague_context_t *dague, 
                          int tree_llvl, int tree_hlvl,
                          int sze_ts, int sze_hlvl,
                          tiled_matrix_desc_t *A, 
                          tiled_matrix_desc_t *TS,
                          tiled_matrix_desc_t *TT) 
{
    dague_object_t *dague_zgeqrf_param = NULL;

    dague_zgeqrf_param = dplasma_zgeqrf_param_New(tree_llvl, tree_hlvl,
                                                  sze_ts, sze_hlvl,
                                                  A, TS, TT);

    dague_enqueue(dague, (dague_object_t*)dague_zgeqrf_param);
    dplasma_progress(dague);

    dplasma_zgeqrf_param_Destruct( dague_zgeqrf_param );
    return 0;
}

void
dplasma_zgeqrf_param_Destruct( dague_object_t *o )
{
    dague_zgeqrf_param_object_t *dague_zgeqrf_param = (dague_zgeqrf_param_object_t *)o;

    dplasma_pivgen_finalize( dague_zgeqrf_param->pivfct );
    dague_private_memory_fini( dague_zgeqrf_param->p_work );
    dague_private_memory_fini( dague_zgeqrf_param->p_tau  );
    free( dague_zgeqrf_param->p_work );
    free( dague_zgeqrf_param->p_tau  );
 
    /* dague_zgeqrf_param_destroy(dague_zgeqrf_param); */
}

