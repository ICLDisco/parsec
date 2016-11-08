/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zgetrf_nopiv.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgetrf_nopiv_New - Generates the handle that computes the LU
 * factorization of a M-by-N matrix A: A = L * U by with no pivoting
 * strategy. The matrix has to be diaagonal dominant to use this
 * routine. Otherwise, the numerical stability of the result is not guaranted.
 *
 * Other variants of LU decomposition with pivoting stragies are available in
 * the library with the following function:
 *     - dplasma_zgetrf_New() that performs LU decomposition with partial pivoting.
 *       This is limited to matrices with column-cyclic distribution.
 *     - dplasma_zgetrf_incpiv_New() that performs tile incremental pivoting
 *       algorithm.
 *     - dplasma_zgetrf_qrf_New() that performs an hybrid LU-QR decomposition.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the factors L and U from the factorization
 *          A = L*U; the unit diagonal elements of L are not stored.
 *
 * @param[out] INFO
 *          On algorithm completion: equal to 0 on success, i if the ith
 *          diagonal value is equal to 0. That implies incoherent result.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague handle describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgetrf_nopiv_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf_nopiv
 * @sa dplasma_zgetrf_nopiv_Destruct
 * @sa dplasma_cgetrf_nopiv_New
 * @sa dplasma_dgetrf_nopiv_New
 * @sa dplasma_sgetrf_nopiv_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zgetrf_nopiv_New( tiled_matrix_desc_t *A,
                          int *INFO )
{
    dague_zgetrf_nopiv_handle_t *dague_getrf_nopiv;

    dague_getrf_nopiv = dague_zgetrf_nopiv_new( A, INFO );

    /* A */
    dplasma_add2arena_tile( dague_getrf_nopiv->arenas[DAGUE_zgetrf_nopiv_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            dague_datatype_double_complex_t, A->mb );

    return (dague_handle_t*)dague_getrf_nopiv;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgetrf_nopiv_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zgetrf_nopiv_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf_nopiv_New
 * @sa dplasma_zgetrf_nopiv
 *
 ******************************************************************************/
void
dplasma_zgetrf_nopiv_Destruct( dague_handle_t *handle )
{
    dague_zgetrf_nopiv_handle_t *dague_zgetrf_nopiv = (dague_zgetrf_nopiv_handle_t *)handle;

    dague_matrix_del2arena( dague_zgetrf_nopiv->arenas[DAGUE_zgetrf_nopiv_DEFAULT_ARENA] );

    dague_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgetrf_nopiv - Computes the LU factorization of a M-by-N matrix A: A
 * = L * U by with no pivoting strategy. The matrix has to be diaagonal dominant
 * to use this routine. Otherwise, the numerical stability of the result is not
 * guaranted.
 *
 * Other variants of LU decomposition with pivoting stragies are available in
 * the library with the following function:
 *     - dplasma_zgetrf_New() that performs LU decomposition with partial pivoting.
 *       This is limited to matrices with column-cyclic distribution.
 *     - dplasma_zgetrf_incpiv_New() that performs tile incremental pivoting
 *       algorithm.
 *     - dplasma_zgetrf_qrf_New() that performs an hybrid LU-QR decomposition.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the factors L and U from the factorization
 *          A = L*U; the unit diagonal elements of L are not stored.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *          \retval i if ith value is singular. Result is incoherent.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf_nopiv
 * @sa dplasma_zgetrf_nopiv_Destruct
 * @sa dplasma_cgetrf_nopiv_New
 * @sa dplasma_dgetrf_nopiv_New
 * @sa dplasma_sgetrf_nopiv_New
 *
 ******************************************************************************/
int
dplasma_zgetrf_nopiv( dague_context_t *dague,
                      tiled_matrix_desc_t *A )
{
    dague_handle_t *dague_zgetrf_nopiv = NULL;

    int info = 0;
    dague_zgetrf_nopiv = dplasma_zgetrf_nopiv_New(A, &info);

    if ( dague_zgetrf_nopiv != NULL ) {
        dague_enqueue( dague, (dague_handle_t*)dague_zgetrf_nopiv);
        dplasma_progress(dague);
        dplasma_zgetrf_nopiv_Destruct( dague_zgetrf_nopiv );
        return info;
    }
    else
        return -101;
}
