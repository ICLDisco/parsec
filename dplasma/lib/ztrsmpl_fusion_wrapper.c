/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "parsec.h"
#include <plasma.h>
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "ztrsmpl_fusion.h"

parsec_handle_t*
dplasma_ztrsmpl_fusion_New( const tiled_matrix_desc_t *A,
                            const tiled_matrix_desc_t *IPIV,
                            tiled_matrix_desc_t *B )
{
    parsec_ztrsmpl_fusion_handle_t *parsec_ztrsmpl_fusion = NULL;
    int nb = A->nb;
    int P = ((two_dim_block_cyclic_t*)A)->grid.rows;

    parsec_ztrsmpl_fusion = parsec_ztrsmpl_fusion_new(A, IPIV, B, P);

    /* A */
    dplasma_add2arena_tile( parsec_ztrsmpl_fusion->arenas[PARSEC_ztrsmpl_fusion_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* PERMUT */
    dplasma_add2arena_rectangle( parsec_ztrsmpl_fusion->arenas[PARSEC_ztrsmpl_fusion_PERMUT_ARENA],
                                 2 * nb * sizeof(int),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_int_t, 2, nb, -1 );

    return (parsec_handle_t*)parsec_ztrsmpl_fusion;
}

void
dplasma_ztrsmpl_fusion_Destruct( parsec_handle_t *handle )
{
    parsec_ztrsmpl_fusion_handle_t *parsec_ztrsmpl_fusion = (parsec_ztrsmpl_fusion_handle_t *)handle;

    parsec_matrix_del2arena( parsec_ztrsmpl_fusion->arenas[PARSEC_ztrsmpl_fusion_DEFAULT_ARENA] );
    parsec_matrix_del2arena( parsec_ztrsmpl_fusion->arenas[PARSEC_ztrsmpl_fusion_PERMUT_ARENA ] );

    parsec_handle_free(handle);
}

void
dplasma_ztrsmpl_fusion( parsec_context_t *parsec,
                        const tiled_matrix_desc_t *A,
                        const tiled_matrix_desc_t *IPIV,
                        tiled_matrix_desc_t *B )
{
    parsec_handle_t *parsec_ztrsmpl_fusion = NULL;

    parsec_ztrsmpl_fusion = dplasma_ztrsmpl_fusion_New(A, IPIV, B );

    if ( parsec_ztrsmpl_fusion != NULL )
    {
        parsec_enqueue( parsec, (parsec_handle_t*)parsec_ztrsmpl_fusion);
        dplasma_progress(parsec);
        dplasma_ztrsmpl_fusion_Destruct( parsec_ztrsmpl_fusion );
    }

    return;
}

