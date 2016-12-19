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
#include "dplasma/lib/dplasmatypes.h"
#include "parsec/private_mempool.h"

#include "zgeqrf.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeqrf_setrecursive - Set the recursive size parameter to enable
 *  recursive DAGs.
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to modify.
 *          On exit, the modified handle.
 *
 * @param[in] hnb
 *          The tile size to use for the smaller recursive call.
 *          hnb must be > 0, otherwise nothing is changed.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_New
 * @sa dplasma_zgeqrf
 *
 ******************************************************************************/
void
dplasma_zgeqrf_setrecursive( parsec_handle_t *handle, int hnb )
{
    parsec_zgeqrf_handle_t *parsec_zgeqrf = (parsec_zgeqrf_handle_t *)handle;

    if (hnb > 0) {
        parsec_zgeqrf->_g_smallnb = hnb;
    }
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeqrf_New - Generates the handle that computes the QR factorization
 * a complex M-by-N matrix A: A = Q * R.
 *
 * The method used in this algorithm is a tile QR algorithm with a flat
 * reduction tree.  It is recommended to use the super tiling parameter (SMB) to
 * improve the performance of the factorization.
 * A high SMB parameter reduces the communication volume, but also deteriorates
 * the load balancing if too important. A small one increases the communication
 * volume, but improves load balancing.
 * A good SMB value should provide enough work to all available cores on one
 * node. It is then recommended to set it to 4 when creating the matrix
 * descriptor.
 * For tiling, MB=200, and IB=32 usually give good results.
 *
 * This variant is good for square large problems.
 * For other problems, see:
 *   - dplasma_zgeqrf_param_New() parameterized with trees for tall and skinny
 *     matrices
 *   - dplasma_zgeqrf_param_New() parameterized with systolic tree if
 *     computation load per node is very low.
 *
 * WARNING: The computations are not done by this call.
 *
 * If you want to enable the recursive DAGs, don't forget to set the recursive
 * tile size and to synchonize the handle ids after the computations since those
 * are for now local. You can follow the code of dplasma_zgeqrf_rec() as an
 * example to do this.
 *
 * Hierarchical DAG Scheduling for Hybrid Distributed Systems; Wu, Wei and
 * Bouteiller, Aurelien and Bosilca, George and Faverge, Mathieu and Dongarra,
 * Jack. 29th IEEE International Parallel & Distributed Processing Symposium,
 * May 2015. (https://hal.inria.fr/hal-0107835)
 *
 *******************************************************************************
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the elements on and above the diagonal of the array contain
 *          the min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular
 *          if (M >= N); the elements below the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          On exit, contains auxiliary information required to compute the Q
 *          matrix, and/or solve the problem.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgeqrf_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf
 * @sa dplasma_zgeqrf_Destruct
 * @sa dplasma_cgeqrf_New
 * @sa dplasma_dgeqrf_New
 * @sa dplasma_sgeqrf_New
 *
 ******************************************************************************/
parsec_handle_t*
dplasma_zgeqrf_New( tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *T )
{
    parsec_zgeqrf_handle_t* handle;
    int ib = T->mb;

    handle = parsec_zgeqrf_new( A,
                               T,
                               ib, NULL, NULL );

    handle->_g_p_tau = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( handle->_g_p_tau, T->nb * sizeof(parsec_complex64_t) );

    handle->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( handle->_g_p_work, ib * T->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( handle->arenas[PARSEC_zgeqrf_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( handle->arenas[PARSEC_zgeqrf_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( handle->arenas[PARSEC_zgeqrf_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 1 );

    /* Little T */
    dplasma_add2arena_rectangle( handle->arenas[PARSEC_zgeqrf_LITTLE_T_ARENA],
                                 T->mb*T->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, T->mb, T->nb, -1);

    return (parsec_handle_t*)handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeqrf_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zgeqrf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_New
 * @sa dplasma_zgeqrf
 *
 ******************************************************************************/
void
dplasma_zgeqrf_Destruct( parsec_handle_t *handle )
{
    parsec_zgeqrf_handle_t *parsec_zgeqrf = (parsec_zgeqrf_handle_t *)handle;

    parsec_matrix_del2arena( parsec_zgeqrf->arenas[PARSEC_zgeqrf_DEFAULT_ARENA   ] );
    parsec_matrix_del2arena( parsec_zgeqrf->arenas[PARSEC_zgeqrf_LOWER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zgeqrf->arenas[PARSEC_zgeqrf_UPPER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zgeqrf->arenas[PARSEC_zgeqrf_LITTLE_T_ARENA  ] );

    parsec_private_memory_fini( parsec_zgeqrf->_g_p_work );
    parsec_private_memory_fini( parsec_zgeqrf->_g_p_tau  );
    free( parsec_zgeqrf->_g_p_work );
    free( parsec_zgeqrf->_g_p_tau  );

    parsec_handle_free(handle);
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeqrf - Computes the QR factorization a M-by-N matrix A:
 * A = Q * R.
 *
 * The method used in this algorithm is a tile QR algorithm with a flat
 * reduction tree. It is recommended to use the super tiling parameter (SMB) to
 * improve the performance of the factorization.
 * A high SMB parameter reduces the communication volume, but also deteriorates
 * the load balancing if too important. A small one increases the communication
 * volume, but improves load balancing.
 * A good SMB value should provide enough work to all available cores on one
 * node. It is then recommended to set it to 4 when creating the matrix
 * descriptor.
 * For tiling, MB=200, and IB=32 usually give good results.
 *
 * This variant is good for square large problems.
 * For other problems, see:
 *   - dplasma_zgeqrf_param() parameterized with trees for tall and skinny
 *     matrices
 *   - dplasma_zgeqrf_param() parameterized with systolic tree if computation
 *     load per node is very low.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the elements on and above the diagonal of the array contain
 *          the min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular
 *          if (M >= N); the elements below the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          On exit, contains auxiliary information required to compute the Q
 *          matrix, and/or solve the problem.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_New
 * @sa dplasma_zgeqrf_Destruct
 * @sa dplasma_cgeqrf
 * @sa dplasma_dgeqrf
 * @sa dplasma_sgeqrf
 *
 ******************************************************************************/
int
dplasma_zgeqrf( parsec_context_t *parsec,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *T )
{
    parsec_handle_t *parsec_zgeqrf = NULL;

    if ( (A->mt != T->mt) || (A->nt != T->nt) ) {
        dplasma_error("dplasma_zgeqrf", "T doesn't have the same number of tiles as A");
        return -101;
    }

    parsec_zgeqrf = dplasma_zgeqrf_New(A, T);

    if ( parsec_zgeqrf != NULL ) {
        parsec_enqueue(parsec, (parsec_handle_t*)parsec_zgeqrf);
        dplasma_progress(parsec);
        dplasma_zgeqrf_Destruct( parsec_zgeqrf );
    }

    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeqrf_rec - Computes the QR factorization a M-by-N matrix A:
 * A = Q * R with recursive DAGs.
 *
 * The method used in this algorithm is a tile QR algorithm with a flat
 * reduction tree. It is recommended to use the super tiling parameter (SMB) to
 * improve the performance of the factorization.
 * A high SMB parameter reduces the communication volume, but also deteriorates
 * the load balancing if too important. A small one increases the communication
 * volume, but improves load balancing.
 * A good SMB value should provide enough work to all available cores on one
 * node. It is then recommended to set it to 4 when creating the matrix
 * descriptor.
 * For tiling, MB=200, and IB=32 usually give good results.
 *
 * This variant is good for square large problems.
 * For other problems, see:
 *   - dplasma_zgeqrf_param() parameterized with trees for tall and skinny
 *     matrices
 *   - dplasma_zgeqrf_param() parameterized with systolic tree if computation
 *     load per node is very low.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the elements on and above the diagonal of the array contain
 *          the min(M,N)-by-N upper trapezoidal matrix R (R is upper triangular
 *          if (M >= N); the elements below the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          On exit, contains auxiliary information required to compute the Q
 *          matrix, and/or solve the problem.
 *
 * @param[in] hnb
 *          The tile size to use for the smaller recursive call.
 *          If hnb <= 0 or hnb > A.nb, the classic algorithm without recursive
 *          calls is applied.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeqrf_New
 * @sa dplasma_zgeqrf_Destruct
 * @sa dplasma_cgeqrf
 * @sa dplasma_dgeqrf
 * @sa dplasma_sgeqrf
 *
 ******************************************************************************/
int
dplasma_zgeqrf_rec( parsec_context_t *parsec,
                    tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *T, int hnb )
{
    parsec_handle_t *parsec_zgeqrf = NULL;

    if ( (A->mt != T->mt) || (A->nt != T->nt) ) {
        dplasma_error("dplasma_zgeqrf", "T doesn't have the same number of tiles as A");
        return -101;
    }

    parsec_zgeqrf = dplasma_zgeqrf_New(A, T);

    if ( parsec_zgeqrf != NULL ) {
        parsec_enqueue(parsec, (parsec_handle_t*)parsec_zgeqrf);
        dplasma_zgeqrf_setrecursive( (parsec_handle_t*)parsec_zgeqrf, hnb );
        dplasma_progress(parsec);
        dplasma_zgeqrf_Destruct( parsec_zgeqrf );
        parsec_handle_sync_ids(); /* recursive DAGs are not synchronous on ids */
    }

    return 0;
}
