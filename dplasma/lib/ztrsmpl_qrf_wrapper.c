/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */


#include <core_blas.h>
#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/dplasmatypes.h"
#include "parsec/private_mempool.h"

#include "ztrsmpl_qrf.h"

parsec_taskpool_t*
dplasma_ztrsmpl_qrf_New( dplasma_qrtree_t *qrtree,
                         tiled_matrix_desc_t *A,
                         tiled_matrix_desc_t *IPIV,
                         tiled_matrix_desc_t *B,
                         tiled_matrix_desc_t *TS,
                         tiled_matrix_desc_t *TT,
                         int *lu_tab )
{
    parsec_ztrsmpl_qrf_taskpool_t* tp;
    int ib = TS->mb;

    /*
     * TODO: We consider ib is T->mb but can be incorrect for some tricks with GPU,
     * it should be passed as a parameter as in getrf
     */

    tp = parsec_ztrsmpl_qrf_new(A,
                                (parsec_ddesc_t*)IPIV,
                                TS,
                                TT,
                                B,
                                lu_tab,
                                *qrtree, ib,
                                NULL, NULL);

    tp->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_p_work, ib * TS->nb * sizeof(parsec_complex64_t) );

    tp->_g_p_tau = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_p_tau, TS->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( tp->arenas[PARSEC_ztrsmpl_qrf_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( tp->arenas[PARSEC_ztrsmpl_qrf_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 1 );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( tp->arenas[PARSEC_ztrsmpl_qrf_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( tp->arenas[PARSEC_ztrsmpl_qrf_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, TS->mb, TS->nb, -1);

    /* IPIV */
    dplasma_add2arena_rectangle( tp->arenas[PARSEC_ztrsmpl_qrf_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_int_t, A->mb, 1, -1 );

    return (parsec_taskpool_t*)tp;
}

void
dplasma_ztrsmpl_qrf_Destruct( parsec_taskpool_t *tp )
{
    parsec_ztrsmpl_qrf_taskpool_t *parsec_ztrsmpl_qrf = (parsec_ztrsmpl_qrf_taskpool_t *)tp;

    parsec_matrix_del2arena( parsec_ztrsmpl_qrf->arenas[PARSEC_ztrsmpl_qrf_DEFAULT_ARENA   ] );
    parsec_matrix_del2arena( parsec_ztrsmpl_qrf->arenas[PARSEC_ztrsmpl_qrf_UPPER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_ztrsmpl_qrf->arenas[PARSEC_ztrsmpl_qrf_LOWER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_ztrsmpl_qrf->arenas[PARSEC_ztrsmpl_qrf_LITTLE_T_ARENA  ] );
    parsec_matrix_del2arena( parsec_ztrsmpl_qrf->arenas[PARSEC_ztrsmpl_qrf_PIVOT_ARENA     ] );

    parsec_private_memory_fini( parsec_ztrsmpl_qrf->_g_p_work );
    parsec_private_memory_fini( parsec_ztrsmpl_qrf->_g_p_tau  );

    free( parsec_ztrsmpl_qrf->_g_p_work );
    free( parsec_ztrsmpl_qrf->_g_p_tau  );

    parsec_taskpool_free(tp);
}

int
dplasma_ztrsmpl_qrf( parsec_context_t *parsec,
                     dplasma_qrtree_t *qrtree,
                     tiled_matrix_desc_t *A,
                     tiled_matrix_desc_t *IPIV,
                     tiled_matrix_desc_t *B,
                     tiled_matrix_desc_t *TS,
                     tiled_matrix_desc_t *TT,
                     int *lu_tab)
{
    parsec_taskpool_t *parsec_ztrsmpl_qrf = NULL;

    parsec_ztrsmpl_qrf = dplasma_ztrsmpl_qrf_New(qrtree, A, IPIV, B, TS, TT, lu_tab);

    parsec_enqueue(parsec, (parsec_taskpool_t*)parsec_ztrsmpl_qrf);
    dplasma_wait_until_completion(parsec);

    dplasma_ztrsmpl_qrf_Destruct( parsec_ztrsmpl_qrf );
    return 0;
}

