/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
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
 *  dplasma_ztrdsm_New - Generates dague handle to compute triangular solve
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
 *          \retval The dague handle describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
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
dague_handle_t*
dplasma_ztrdsm_New(const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B )
{
    dague_handle_t *dague_trdsm = NULL; 

    dague_trdsm = (dague_handle_t*)dague_ztrdsm_new( *A, (dague_ddesc_t*)A,
                                                     *B, (dague_ddesc_t*)B );

    dplasma_add2arena_tile(((dague_ztrdsm_handle_t*)dague_trdsm)->arenas[DAGUE_ztrdsm_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           dague_datatype_double_complex_t, A->mb);

    return dague_trdsm;
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
dplasma_ztrdsm_Destruct( dague_handle_t *handle )
{
    dague_ztrdsm_handle_t *otrdsm = (dague_ztrdsm_handle_t *)handle;
    dague_matrix_del2arena( otrdsm->arenas[DAGUE_ztrdsm_DEFAULT_ARENA] );
    handle->destructor(handle);
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
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
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
dplasma_ztrdsm( dague_context_t *dague,
                const tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *B)
{
    dague_handle_t *dague_ztrdsm = NULL;

    dague_ztrdsm = dplasma_ztrdsm_New(A, B);

    if ( dague_ztrdsm != NULL ) {
        dague_enqueue( dague, dague_ztrdsm );
        dplasma_progress( dague );

        dplasma_ztrdsm_Destruct( dague_ztrdsm );
    }

    return 0;
}
