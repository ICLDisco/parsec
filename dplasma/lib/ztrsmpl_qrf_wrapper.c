/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */


#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dague/private_mempool.h"

#include "ztrsmpl_qrf.h"

dague_handle_t*
dplasma_ztrsmpl_qrf_New( dplasma_qrtree_t *qrtree,
                         tiled_matrix_desc_t *A,
                         tiled_matrix_desc_t *IPIV,
                         tiled_matrix_desc_t *B,
                         tiled_matrix_desc_t *TS,
                         tiled_matrix_desc_t *TT,
                         int *lu_tab )
{
    dague_ztrsmpl_qrf_handle_t* handle;
    int ib = TS->mb;

    /*
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf
     */

    handle = dague_ztrsmpl_qrf_new((dague_ddesc_t*)A,
                                   (dague_ddesc_t*)IPIV,
                                   (dague_ddesc_t*)TS,
                                   (dague_ddesc_t*)TT,
                                   (dague_ddesc_t*)B,
                                   lu_tab,
                                   *qrtree, ib,
                                   NULL, NULL);

    handle->_g_p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( handle->_g_p_work, ib * TS->nb * sizeof(dague_complex64_t) );

    handle->_g_p_tau = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( handle->_g_p_tau, TS->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( handle->arenas[DAGUE_ztrsmpl_qrf_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            dague_datatype_double_complex_t, A->mb );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( handle->arenas[DAGUE_ztrsmpl_qrf_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             dague_datatype_double_complex_t, A->mb, 1 );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( handle->arenas[DAGUE_ztrsmpl_qrf_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             dague_datatype_double_complex_t, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( handle->arenas[DAGUE_ztrsmpl_qrf_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_double_complex_t, TS->mb, TS->nb, -1);

    /* IPIV */
    dplasma_add2arena_rectangle( handle->arenas[DAGUE_ztrsmpl_qrf_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_int_t, A->mb, 1, -1 );

    return (dague_handle_t*)handle;
}

void
dplasma_ztrsmpl_qrf_Destruct( dague_handle_t *handle )
{
    dague_ztrsmpl_qrf_handle_t *dague_ztrsmpl_qrf = (dague_ztrsmpl_qrf_handle_t *)handle;

    dague_matrix_del2arena( dague_ztrsmpl_qrf->arenas[DAGUE_ztrsmpl_qrf_DEFAULT_ARENA   ] );
    dague_matrix_del2arena( dague_ztrsmpl_qrf->arenas[DAGUE_ztrsmpl_qrf_UPPER_TILE_ARENA] );
    dague_matrix_del2arena( dague_ztrsmpl_qrf->arenas[DAGUE_ztrsmpl_qrf_LOWER_TILE_ARENA] );
    dague_matrix_del2arena( dague_ztrsmpl_qrf->arenas[DAGUE_ztrsmpl_qrf_LITTLE_T_ARENA  ] );
    dague_matrix_del2arena( dague_ztrsmpl_qrf->arenas[DAGUE_ztrsmpl_qrf_PIVOT_ARENA     ] );

    dague_private_memory_fini( dague_ztrsmpl_qrf->_g_p_work );
    dague_private_memory_fini( dague_ztrsmpl_qrf->_g_p_tau  );

    free( dague_ztrsmpl_qrf->_g_p_work );
    free( dague_ztrsmpl_qrf->_g_p_tau  );

    dague_handle_free(handle);
}

int
dplasma_ztrsmpl_qrf( dague_context_t *dague,
                     dplasma_qrtree_t *qrtree,
                     tiled_matrix_desc_t *A,
                     tiled_matrix_desc_t *IPIV,
                     tiled_matrix_desc_t *B,
                     tiled_matrix_desc_t *TS,
                     tiled_matrix_desc_t *TT,
                     int *lu_tab)
{
    dague_handle_t *dague_ztrsmpl_qrf = NULL;

    dague_ztrsmpl_qrf = dplasma_ztrsmpl_qrf_New(qrtree, A, IPIV, B, TS, TT, lu_tab);

    dague_enqueue(dague, (dague_handle_t*)dague_ztrsmpl_qrf);
    dplasma_progress(dague);

    dplasma_ztrsmpl_qrf_Destruct( dague_ztrsmpl_qrf );
    return 0;
}

