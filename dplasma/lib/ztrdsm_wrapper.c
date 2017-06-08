/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */


#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "ztrdsm.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztrdsm_New - Generates parsec handle to compute triangular solve
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
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
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
parsec_handle_t*
dplasma_ztrdsm_New(const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B )
{
    parsec_handle_t *parsec_trdsm = NULL; 

    parsec_trdsm = (parsec_handle_t*)parsec_ztrdsm_new( A, B );

    dplasma_add2arena_tile(((parsec_ztrdsm_handle_t*)parsec_trdsm)->arenas[PARSEC_ztrdsm_DEFAULT_ARENA],
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
 *  dplasma_ztrdsm_Destruct - Free the data structure associated to an handle
 *  created with dplasma_ztrdsm_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrdsm_New
 * @sa dplasma_ztrdsm
 *
 ******************************************************************************/
void
dplasma_ztrdsm_Destruct( parsec_handle_t *handle )
{
    parsec_ztrdsm_handle_t *otrdsm = (parsec_ztrdsm_handle_t *)handle;
    parsec_matrix_del2arena( otrdsm->arenas[PARSEC_ztrdsm_DEFAULT_ARENA] );
    parsec_handle_free(handle);
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
                const tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *B)
{
    parsec_handle_t *parsec_ztrdsm = NULL;

    parsec_ztrdsm = dplasma_ztrdsm_New(A, B);

    if ( parsec_ztrdsm != NULL ) {
        parsec_enqueue( parsec, parsec_ztrdsm );
        dplasma_progress( parsec );

        dplasma_ztrdsm_Destruct( parsec_ztrdsm );
    }

    return 0;
}
