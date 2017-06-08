/*
 * Copyright (c) 2011-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "parsec/vpmap.h"
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
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
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
parsec_handle_t*
dplasma_zgetrf_New( tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *IPIV,
                    int *INFO )
{
    parsec_zgetrf_handle_t *parsec_getrf;
    int nbthreads = dplasma_imax( 1, vpmap_get_nb_threads_in_vp(0) - 1 );

    if ( (IPIV->mt != 1) || (dplasma_imin(A->nt, A->mt) > IPIV->nt)) {
        dplasma_error("dplasma_zgetrf_New", "IPIV doesn't have the correct number of tiles (1-by-min(A->mt,A->nt)");
        return NULL;
    }

    parsec_getrf = parsec_zgetrf_new( A,
                                    (parsec_ddesc_t*)IPIV,
                                    INFO );

#if defined(CORE_GETRF_270)

    if ( A->storage == matrix_Tile ) {
        CORE_zgetrf_rectil_init();
    } else {
        CORE_zgetrf_reclap_init();
    }
    parsec_getrf->_g_nbmaxthrd = dplasma_imin( nbthreads, 48 );

#else

    if ( A->storage == matrix_Tile ) {
        parsec_getrf->_g_getrfdata = CORE_zgetrf_rectil_init(nbthreads);
    } else {
        parsec_getrf->_g_getrfdata = CORE_zgetrf_reclap_init(nbthreads);
    }
    parsec_getrf->_g_nbmaxthrd = nbthreads;

#endif

    /* A */
    dplasma_add2arena_tile( parsec_getrf->arenas[PARSEC_zgetrf_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* IPIV */
    dplasma_add2arena_rectangle( parsec_getrf->arenas[PARSEC_zgetrf_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_int_t, 1, A->mb, -1 );

    return (parsec_handle_t*)parsec_getrf;
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
dplasma_zgetrf_Destruct( parsec_handle_t *handle )
{
    parsec_zgetrf_handle_t *parsec_zgetrf = (parsec_zgetrf_handle_t *)handle;

    parsec_matrix_del2arena( parsec_zgetrf->arenas[PARSEC_zgetrf_DEFAULT_ARENA] );
    parsec_matrix_del2arena( parsec_zgetrf->arenas[PARSEC_zgetrf_PIVOT_ARENA  ] );

    if ( parsec_zgetrf->_g_getrfdata != NULL )
        free( parsec_zgetrf->_g_getrfdata );

    parsec_handle_free(handle);
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
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
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
dplasma_zgetrf( parsec_context_t *parsec,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *IPIV )
{
    parsec_handle_t *parsec_zgetrf = NULL;

    int info = 0;

    if ( (IPIV->mt != 1) || (dplasma_imin(A->nt, A->mt) > IPIV->nt)) {
        dplasma_error("dplasma_zgetrf", "IPIV doesn't have the correct number of tiles (1-by-min(A->mt,A->nt)");
        return -3;
    }

    parsec_zgetrf = dplasma_zgetrf_New(A, IPIV, &info);

    if ( parsec_zgetrf != NULL ) {
        parsec_enqueue( parsec, parsec_zgetrf );
        dplasma_progress(parsec);
        dplasma_zgetrf_Destruct( parsec_zgetrf );
        return info;
    }
    else
        return -101;
}
