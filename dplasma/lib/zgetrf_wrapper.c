/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dague/vpmap.h"
#include "dplasma/lib/dplasmajdf.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zgetrf.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgetrf_New - Generates the handle that computes the LU factorization
 * of a M-by-N matrix A: A = P * L * U by partial pivoting algorithm.
 *
 * This algorithm exploits the multi-threaded recursive kernels of the PLASMA
 * library and by consequence require a column-cyclic data distribution if used
 * in distributed memory.
 * This is not an optimal solution for distributed memory system, and should be
 * used only if no other possibiliies is available. Absolute priority scheduler
 * is known to improve the performance of this algorithm and should be prefered.
 *
 * Other variants of LU decomposition are available in the library wioth the
 * following function:
 *     - dplasma_zgetrf_incpiv_New() that performs tile incremental pivoting
 *       algorithm.
 *     - dplasma_zgetrf_nopiv_New() that performs LU decomposition with no pivoting
 *       if the matrix is known as beeing diagonal dominant.
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
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 * @param[out] IPIV
 *          Descriptor of the IPIV matrix. Should be of size 1-by-min(M,N).
 *          On exit, contains the pivot indices; for 1 <= i <= min(M,N), row i
 *          of the matrix was interchanged with row IPIV(i).
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
 *          destroy with dplasma_zgetrf_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf
 * @sa dplasma_zgetrf_Destruct
 * @sa dplasma_cgetrf_New
 * @sa dplasma_dgetrf_New
 * @sa dplasma_sgetrf_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zgetrf_New( tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *IPIV,
                    int *INFO )
{
    dague_zgetrf_handle_t *dague_getrf;
    int nbthreads = dplasma_imax( 1, vpmap_get_nb_threads_in_vp(0) - 1 );

    if ( (IPIV->mt != 1) || (dplasma_imin(A->nt, A->mt) > IPIV->nt)) {
        dplasma_error("dplasma_zgetrf_New", "IPIV doesn't have the correct number of tiles (1-by-min(A->mt,A->nt)");
        return NULL;
    }

    dague_getrf = dague_zgetrf_new( (dague_ddesc_t*)A,
                                    (dague_ddesc_t*)IPIV,
                                    INFO );

#if defined(CORE_GETRF_270)

    if ( A->storage == matrix_Tile ) {
        CORE_zgetrf_rectil_init();
    } else {
        CORE_zgetrf_reclap_init();
    }
    dague_getrf->nbmaxthrd = dplasma_imin( nbthreads, 48 );

#else

    if ( A->storage == matrix_Tile ) {
        dague_getrf->getrfdata = CORE_zgetrf_rectil_init(nbthreads);
    } else {
        dague_getrf->getrfdata = CORE_zgetrf_reclap_init(nbthreads);
    }
    dague_getrf->nbmaxthrd = nbthreads;

#endif

    /* A */
    dplasma_add2arena_tile( dague_getrf->arenas[DAGUE_zgetrf_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            dague_datatype_double_complex_t, A->mb );

    /* IPIV */
    dplasma_add2arena_rectangle( dague_getrf->arenas[DAGUE_zgetrf_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_int_t, 1, A->mb, -1 );

    return (dague_handle_t*)dague_getrf;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgetrf_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zgetrf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgetrf_New
 * @sa dplasma_zgetrf
 *
 ******************************************************************************/
void
dplasma_zgetrf_Destruct( dague_handle_t *handle )
{
    dague_zgetrf_handle_t *dague_zgetrf = (dague_zgetrf_handle_t *)handle;

    dague_matrix_del2arena( dague_zgetrf->arenas[DAGUE_zgetrf_DEFAULT_ARENA] );
    dague_matrix_del2arena( dague_zgetrf->arenas[DAGUE_zgetrf_PIVOT_ARENA  ] );

    if ( dague_zgetrf->getrfdata != NULL )
        free( dague_zgetrf->getrfdata );

    handle->destructor(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgetrf - Computes the LU factorization of a M-by-N matrix A: A = P *
 * L * U by partial pivoting algorithm.
 *
 * This algorithm exploits the multi-threaded recursive kernels of the PLASMA
 * library and by consequence require a column-cyclic data distribution if used
 * in distributed memory.
 * This is not an optimal solution for distributed memory system, and should be
 * used only if no other possibiliies is available. Absolute priority scheduler
 * is known to improve the performance of this algorithm and should be prefered.
 *
 * Other variants of LU decomposition are available in the library wioth the
 * following function:
 *     - dplasma_zgetrf_incpiv() that performs tile incremental pivoting
 *       algorithm.
 *     - dplasma_zgetrf_nopiv() that performs LU decomposition with no pivoting
 *       if the matrix is known as beeing diagonal dominant.
 *     - dplasma_zgetrf_qrf() that performs an hybrid LU-QR decomposition.
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
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 * @param[out] IPIV
 *          Descriptor of the IPIV matrix. Should be of size 1-by-min(M,N).
 *          On exit, contains the pivot indices; for 1 <= i <= min(M,N), row i
 *          of the matrix was interchanged with row IPIV(i).
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
 * @sa dplasma_zgetrf
 * @sa dplasma_zgetrf_Destruct
 * @sa dplasma_cgetrf_New
 * @sa dplasma_dgetrf_New
 * @sa dplasma_sgetrf_New
 *
 ******************************************************************************/
int
dplasma_zgetrf( dague_context_t *dague,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *IPIV )
{
    dague_handle_t *dague_zgetrf = NULL;

    int info = 0;

    if ( (IPIV->mt != 1) || (dplasma_imin(A->nt, A->mt) > IPIV->nt)) {
        dplasma_error("dplasma_zgetrf", "IPIV doesn't have the correct number of tiles (1-by-min(A->mt,A->nt)");
        return -3;
    }

    dague_zgetrf = dplasma_zgetrf_New(A, IPIV, &info);

    if ( dague_zgetrf != NULL ) {
        dague_enqueue( dague, dague_zgetrf );
        dplasma_progress(dague);
        dplasma_zgetrf_Destruct( dague_zgetrf );
        return info;
    }
    else
        return -101;
}
