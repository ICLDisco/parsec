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
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "parsec/private_mempool.h"

#include "zgelqf.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgelqf_New - Generates the taskpool that computes the LQ factorization
 * a complex M-by-N matrix A: A = L * Q.
 *
 * The method used in this algorithm is a tile LQ algorithm with a flat
 * reduction tree. It is recommended to use the super tiling parameter (SNB) to
 * improve the performance of the factorization.
 * A high SNB parameter reduces the communication volume, but also deteriorates
 * the load balancing if too important. A small one increases the communication
 * volume, but improves load balancing.
 * A good SNB value should provide enough work to all available cores on one
 * node. It is then recommended to set it to 4 when creating the matrix
 * descriptor.
 * For tiling, MB=200, and IB=32 usually give good results.
 *
 * This variant is good for square large problems.
 * For other problems, see:
 *   - dplasma_zgelqf_param_New() parameterized with trees for tall and skinny
 *     matrices
 *   - dplasma_zgelqf_param_New() parameterized with systolic tree if
 *     computation load per node is very low.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A to be factorized.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the elements on and below the diagonal of the array contain
 *          the M-by-min(M,N) lower trapezoidal matrix L (L is lower triangular
 *          if (N >= M); the elements above the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile LQ algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          On exit, contains auxiliary information required to compute the Q
 *          matrix, and/or solve the problem.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgelqf_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgelqf
 * @sa dplasma_zgelqf_Destruct
 * @sa dplasma_cgelqf_New
 * @sa dplasma_dgelqf_New
 * @sa dplasma_sgelqf_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zgelqf_New( parsec_tiled_matrix_dc_t *A,
                    parsec_tiled_matrix_dc_t *T )
{
    parsec_zgelqf_taskpool_t* tp;
    int ib = T->mb;

    tp = parsec_zgelqf_new( A,
                            T,
                            ib, NULL, NULL );

    tp->_g_p_tau = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_p_tau, T->nb * sizeof(parsec_complex64_t) );

    tp->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( tp->_g_p_work, ib * T->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( tp->arenas[PARSEC_zgelqf_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* Lower triangular part of tile with diagonal */
    dplasma_add2arena_lower( tp->arenas[PARSEC_zgelqf_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 1 );

    /* Upper triangular part of tile without diagonal */
    dplasma_add2arena_upper( tp->arenas[PARSEC_zgelqf_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( tp->arenas[PARSEC_zgelqf_LITTLE_T_ARENA],
                                 T->mb*T->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, T->mb, T->nb, -1);

    return (parsec_taskpool_t*)tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgelqf_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zgelqf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgelqf_New
 * @sa dplasma_zgelqf
 *
 ******************************************************************************/
void
dplasma_zgelqf_Destruct( parsec_taskpool_t *tp )
{
    parsec_zgelqf_taskpool_t *parsec_zgelqf = (parsec_zgelqf_taskpool_t *)tp;

    parsec_matrix_del2arena( parsec_zgelqf->arenas[PARSEC_zgelqf_DEFAULT_ARENA   ] );
    parsec_matrix_del2arena( parsec_zgelqf->arenas[PARSEC_zgelqf_LOWER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zgelqf->arenas[PARSEC_zgelqf_UPPER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zgelqf->arenas[PARSEC_zgelqf_LITTLE_T_ARENA  ] );

    parsec_private_memory_fini( parsec_zgelqf->_g_p_work );
    parsec_private_memory_fini( parsec_zgelqf->_g_p_tau  );
    free( parsec_zgelqf->_g_p_work );
    free( parsec_zgelqf->_g_p_tau  );

    parsec_taskpool_free(tp);
}


/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgelqf - Computes the LQ factorization a M-by-N matrix A:
 * A = L * Q.
 *
 * The method used in this algorithm is a tile LQ algorithm with a flat
 * reduction tree. It is recommended to use the super tiling parameter (SNB) to
 * improve the performance of the factorization.
 * A high SNB parameter reduces the communication volume, but also deteriorates
 * the load balancing if too important. A small one increases the communication
 * volume, but improves load balancing.
 * A good SNB value should provide enough work to all available cores on one
 * node. It is then recommended to set it to 4 when creating the matrix
 * descriptor.
 *
 * This variant is good for square large problems.
 * For other problems, see:
 *   - dplasma_zgelqf_param() parameterized with trees for tall and skinny
 *     matrices
 *   - dplasma_zgelqf_param() parameterized with systolic tree if computation
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
 *          On exit, the elements on and below the diagonal of the array contain
 *          the M-by-min(M,N) lower trapezoidal matrix L (L is lower triangular
 *          if (N >= M); the elements above the diagonal represent the unitary
 *          matrix Q as a product of elementary reflectors stored by tiles.
 *          It cannot be used directly as in Lapack.
 *
 * @param[out] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile LQ algorithm. This matrix must be
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
 * @sa dplasma_zgelqf_New
 * @sa dplasma_zgelqf_Destruct
 * @sa dplasma_cgelqf
 * @sa dplasma_dgelqf
 * @sa dplasma_sgelqf
 *
 ******************************************************************************/
int
dplasma_zgelqf( parsec_context_t *parsec,
                parsec_tiled_matrix_dc_t *A,
                parsec_tiled_matrix_dc_t *T )
{
    parsec_taskpool_t *parsec_zgelqf = NULL;

    if ( (A->mt != T->mt) || (A->nt != T->nt) ) {
        dplasma_error("dplasma_zgelqf", "T doesn't have the same number of tiles as A");
        return -101;
    }

    parsec_zgelqf = dplasma_zgelqf_New(A, T);

    if ( parsec_zgelqf != NULL ) {
        parsec_enqueue(parsec, (parsec_taskpool_t*)parsec_zgelqf);
        dplasma_wait_until_completion(parsec);
        dplasma_zgelqf_Destruct( parsec_zgelqf );
    }

    return 0;
}
