/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */

#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zlaswp.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlaswp_New - Generates the handle that performs a series of row
 *  interchanges on the matrix A.  One row interchange is initiated for each
 *  rows in IPIV descriptor.
 *
 *  WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the matrix with row interchanges applied.
 *
 * @param[in] IPIV
 *          Descriptor of pivot array IPIV that contains the row interchanges.
 *
 * @param[in] inc
 *          Order in which row interchanges are applied.
 *          If 1, starts from the beginning.
 *          If -1, starts from the end.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zlaswp_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zlaswp
 * @sa dplasma_zlaswp_Destruct
 * @sa dplasma_claswp_New
 * @sa dplasma_dlaswp_New
 * @sa dplasma_slaswp_New
 *
 ******************************************************************************/
parsec_handle_t *
dplasma_zlaswp_New(tiled_matrix_desc_t *A,
                   const tiled_matrix_desc_t *IPIV,
                   int inc)
{
    parsec_zlaswp_handle_t *parsec_laswp;

    parsec_laswp = parsec_zlaswp_new( A,
                                    IPIV,
                                    inc );

    /* A */
    dplasma_add2arena_tile( parsec_laswp->arenas[PARSEC_zlaswp_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* IPIV */
    dplasma_add2arena_rectangle( parsec_laswp->arenas[PARSEC_zlaswp_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_int_t, 1, A->mb, -1 );

    return (parsec_handle_t*)parsec_laswp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlaswp_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zlaswp_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlaswp_New
 * @sa dplasma_zlaswp
 *
 ******************************************************************************/
void
dplasma_zlaswp_Destruct( parsec_handle_t *handle )
{
    parsec_zlaswp_handle_t *parsec_zlaswp = (parsec_zlaswp_handle_t *)handle;

    parsec_matrix_del2arena( parsec_zlaswp->arenas[PARSEC_zlaswp_DEFAULT_ARENA] );
    parsec_matrix_del2arena( parsec_zlaswp->arenas[PARSEC_zlaswp_PIVOT_ARENA  ] );

    parsec_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zlaswp - Performs a series of row interchanges on the matrix A.  One
 *  row interchange is initiated for each rows in IPIV descriptor.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the matrix with row interchanges applied.
 *
 * @param[in] IPIV
 *          Descriptor of pivot array IPIV that contains the row interchanges.
 *
 * @param[in] inc
 *          Order in which row interchanges are applied.
 *          If 1, starts from the beginning.
 *          If -1, starts from the end.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zlaswp_New
 * @sa dplasma_zlaswp_Destruct
 * @sa dplasma_claswp
 * @sa dplasma_dlaswp
 * @sa dplasma_slaswp
 *
 ******************************************************************************/
int
dplasma_zlaswp( parsec_context_t *parsec,
                tiled_matrix_desc_t *A,
                const tiled_matrix_desc_t *IPIV,
                int inc)
{
    parsec_handle_t *parsec_zlaswp = NULL;

    parsec_zlaswp = dplasma_zlaswp_New(A, IPIV, inc);

    parsec_enqueue( parsec, (parsec_handle_t*)parsec_zlaswp);
    dplasma_wait_until_completion(parsec);

    dplasma_zlaswp_Destruct( parsec_zlaswp );

    return 0;
}
