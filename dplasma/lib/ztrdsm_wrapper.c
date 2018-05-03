/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */


#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "ztrdsm.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztrdsm_New - Generates parsec taskpool to compute triangular solve
 *     op( A ) * X = B or X * op( A ) = B
 *  WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] A
 *
 * @param[in,out] B
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_ztrdsm_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrdsm
 * @sa dplasma_ztrdsm_Destruct
 * @sa dplasma_ctrdsm_New
 * @sa dplasma_dtrdsm_New
 * @sa dplasma_strdsm_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_ztrdsm_New(const parsec_tiled_matrix_dc_t *A, parsec_tiled_matrix_dc_t *B )
{
    parsec_taskpool_t *parsec_trdsm = NULL;

    parsec_trdsm = (parsec_taskpool_t*)parsec_ztrdsm_new( A, B );

    dplasma_add2arena_tile(((parsec_ztrdsm_taskpool_t*)parsec_trdsm)->arenas[PARSEC_ztrdsm_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_double_complex_t, A->mb);

    return parsec_trdsm;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztrdsm_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_ztrdsm_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrdsm_New
 * @sa dplasma_ztrdsm
 *
 ******************************************************************************/
void
dplasma_ztrdsm_Destruct( parsec_taskpool_t *tp )
{
    parsec_ztrdsm_taskpool_t *otrdsm = (parsec_ztrdsm_taskpool_t *)tp;
    parsec_matrix_del2arena( otrdsm->arenas[PARSEC_ztrdsm_DEFAULT_ARENA] );
    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztrdsm -
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] A
 *
 * @param[in,out] B
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrdsm_New
 * @sa dplasma_ztrdsm_Destruct
 * @sa dplasma_ctrdsm
 * @sa dplasma_dtrdsm
 * @sa dplasma_strdsm
 *
 ******************************************************************************/
int
dplasma_ztrdsm( parsec_context_t *parsec,
                const parsec_tiled_matrix_dc_t *A,
                parsec_tiled_matrix_dc_t *B)
{
    parsec_taskpool_t *parsec_ztrdsm = NULL;

    parsec_ztrdsm = dplasma_ztrdsm_New(A, B);

    if ( parsec_ztrdsm != NULL ) {
        parsec_context_add_taskpool( parsec, parsec_ztrdsm );
        dplasma_wait_until_completion( parsec );

        dplasma_ztrdsm_Destruct( parsec_ztrdsm );
    }

    return 0;
}
