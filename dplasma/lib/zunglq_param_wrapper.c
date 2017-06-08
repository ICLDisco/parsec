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

#include "zunglq_param.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunglq_param_New - Generates the parsec handle that computes the generation
 *  of an M-by-N matrix Q with orthonormal rows, which is defined as the
 *  first M rows of a product of K elementary reflectors of order N
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgelqf_param_New().
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] qrtree
 *          The structure that describes the trees used to perform the
 *          hierarchical QR factorization.
 *          See dplasma_hqr_init() or dplasma_systolic_init().
 *
 * @param[in] A
 *          Descriptor of the matrix A of size K-by-N factorized with the
 *          dplasma_zgelqf_param_New() routine.
 *          On entry, the i-th row must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgelqf_param_New() in the first k rows of its array
 *          argument A. M >= K >= 0.
 *
 * @param[in] TS
 *          Descriptor of the matrix TS distributed exactly as the A
 *          matrix. TS.mb defines the IB parameter of tile LQ algorithm. This
 *          matrix must be of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgelqf_param_New().
 *
 * @param[in] TT
 *          Descriptor of the matrix TT distributed exactly as the A
 *          matrix. TT.mb defines the IB parameter of tile LQ algorithm. This
 *          matrix must be of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgelqf_param_New().
 *
 * @param[in,out] Q
 *          Descriptor of the M-by-N matrix Q with orthonormal rows.
 *          On entry, the Id matrix.
 *          On exit, the orthonormal matrix Q.
 *          N >= M >= 0.
 *
 *******************************************************************************
 *
 * @return
 *          \retval The parsec handle which describes the operation to perform
 *                  NULL if one of the parameter is incorrect
 *
 *******************************************************************************
 *
 * @sa dplasma_zunglq_param_Destruct
 * @sa dplasma_zunglq_param
 * @sa dplasma_cunglq_param_New
 * @sa dplasma_dorglq_param_New
 * @sa dplasma_sorglq_param_New
 * @sa dplasma_zgelqf_param_New
 *
 ******************************************************************************/
parsec_handle_t*
dplasma_zunglq_param_New( dplasma_qrtree_t *qrtree,
                          tiled_matrix_desc_t *A,
                          tiled_matrix_desc_t *TS,
                          tiled_matrix_desc_t *TT,
                          tiled_matrix_desc_t *Q )
{
    parsec_zunglq_param_handle_t* handle;
    int ib = TS->mb;

    if ( Q->m > Q->n ) {
        dplasma_error("dplasma_zunglq_param_New", "illegal size of Q (M should be smaller or equal to N)");
        return NULL;
    }
    if ( A->m > Q->m ) {
        dplasma_error("dplasma_zunglq_param_New", "illegal size of A (K should be smaller or equal to M)");
        return NULL;
    }
    if ( (TS->nt < A->nt) || (TS->mt < A->mt) ) {
        dplasma_error("dplasma_zunglq_param_New", "illegal size of TS (TS should have as many tiles as A)");
        return NULL;
    }
    if ( (TT->nt < A->nt) || (TT->mt < A->mt) ) {
        dplasma_error("dplasma_zunglq_param_New", "illegal size of TT (TT should have as many tiles as A)");
        return NULL;
    }

    handle = parsec_zunglq_param_new( A,
                                     TS,
                                     TT,
                                     Q,
                                     *qrtree,
                                     NULL );

    handle->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( handle->_g_p_work, ib * TS->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( handle->arenas[PARSEC_zunglq_param_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( handle->arenas[PARSEC_zunglq_param_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( handle->arenas[PARSEC_zunglq_param_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 1 );

    /* Little T */
    dplasma_add2arena_rectangle( handle->arenas[PARSEC_zunglq_param_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, TS->mb, TS->nb, -1);

    return (parsec_handle_t*)handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunglq_param_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zunglq_param_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunglq_param_New
 * @sa dplasma_zunglq_param
 *
 ******************************************************************************/
void
dplasma_zunglq_param_Destruct( parsec_handle_t *handle )
{
    parsec_zunglq_param_handle_t *parsec_zunglq = (parsec_zunglq_param_handle_t *)handle;

    parsec_matrix_del2arena( parsec_zunglq->arenas[PARSEC_zunglq_param_DEFAULT_ARENA   ] );
    parsec_matrix_del2arena( parsec_zunglq->arenas[PARSEC_zunglq_param_UPPER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zunglq->arenas[PARSEC_zunglq_param_LOWER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zunglq->arenas[PARSEC_zunglq_param_LITTLE_T_ARENA  ] );

    parsec_private_memory_fini( parsec_zunglq->_g_p_work );
    free( parsec_zunglq->_g_p_work );

    parsec_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunglq_param - Generates of an M-by-N matrix Q with orthonormal rows,
 *  which is defined as the first M rows of a product of K elementary
 *  reflectors of order N
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgelqf_param_New().
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] qrtree
 *          The structure that describes the trees used to perform the
 *          hierarchical QR factorization.
 *          See dplasma_hqr_init() or dplasma_systolic_init().
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgelqf_param_New() routine.
 *          On entry, the i-th row must contain the vector which defines the
 *          elementary reflector H(i), for i = 1,2,...,k, as returned by
 *          dplasma_zgelqf_param_New() in the first k rows of its array argument
 *          A. M >= K >= 0.
 *
 * @param[in] TS
 *          Descriptor of the matrix TS distributed exactly as the A
 *          matrix. TS.mb defines the IB parameter of tile LQ algorithm. This
 *          matrix must be of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgelqf_param_New().
 *
 * @param[in] TT
 *          Descriptor of the matrix TT distributed exactly as the A
 *          matrix. TT.mb defines the IB parameter of tile LQ algorithm. This
 *          matrix must be of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgelqf_param_New().
 *
 * @param[out] Q
 *          Descriptor of the M-by-N matrix Q with orthonormal rows.
 *          On entry, the Id matrix.
 *          On exit, the orthonormal matrix Q.
 *          N >= M >= 0.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunglq_param_New
 * @sa dplasma_zunglq_param_Destruct
 * @sa dplasma_cunglq_param
 * @sa dplasma_dorglq_param
 * @sa dplasma_sorglq_param
 * @sa dplasma_zgelqf_param
 *
 ******************************************************************************/
int
dplasma_zunglq_param( parsec_context_t *parsec,
                      dplasma_qrtree_t *qrtree,
                      tiled_matrix_desc_t *A,
                      tiled_matrix_desc_t *TS,
                      tiled_matrix_desc_t *TT,
                      tiled_matrix_desc_t *Q )
{
    parsec_handle_t *parsec_zunglq;

    if (parsec == NULL) {
        dplasma_error("dplasma_zunglq_param", "dplasma not initialized");
        return -1;
    }
    if ( Q->m > Q->n) {
        dplasma_error("dplasma_zunglq_param", "illegal number of rows in Q (M)");
        return -2;
    }
    if ( A->m > Q->m) {
        dplasma_error("dplasma_zunglq_param", "illegal number of rows in A (K)");
        return -3;
    }
    if ( A->n != Q->n ) {
        dplasma_error("dplasma_zunglq_param", "illegal number of columns in A");
        return -5;
    }

    if (dplasma_imin(Q->m, dplasma_imin(Q->n, A->m)) == 0)
        return 0;

    dplasma_qrtree_check( A, qrtree );

    parsec_zunglq = dplasma_zunglq_param_New(qrtree, A, TS, TT, Q);

    if ( parsec_zunglq != NULL ){
        parsec_enqueue(parsec, parsec_zunglq);
        dplasma_progress(parsec);
        dplasma_zunglq_param_Destruct( parsec_zunglq );
    }
    return 0;
}
