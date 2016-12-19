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
#include "dplasma/lib/dplasmaaux.h"
#include "parsec/private_mempool.h"

#include "zunglq.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunglq_New - Generates the parsec handle that computes the generation
 *  of an M-by-N matrix Q with orthonormal rows, which is defined as the
 *  first M rows of a product of K elementary reflectors of order N
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgelqf_New().
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] A
 *          Descriptor of the matrix A of size K-by-N factorized with the
 *          dplasma_zgelqf_New() routine.
 *          On entry, the i-th row must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgelqf_New() in the first k rows of its array
 *          argument A. M >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile LQ algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgelqf_New().
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
 * @sa dplasma_zunglq_Destruct
 * @sa dplasma_zunglq
 * @sa dplasma_cunglq_New
 * @sa dplasma_dorglq_New
 * @sa dplasma_sorglq_New
 * @sa dplasma_zgelqf_New
 *
 ******************************************************************************/
parsec_handle_t*
dplasma_zunglq_New( tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *T,
                    tiled_matrix_desc_t *Q )
{
    parsec_zunglq_handle_t* handle;
    int ib = T->mb;

    if ( Q->m > Q->n ) {
        dplasma_error("dplasma_zunglq_New", "illegal size of Q (M should be smaller or equal to N)");
        return NULL;
    }
    if ( A->m > Q->m ) {
        dplasma_error("dplasma_zunglq_New", "illegal size of A (K should be smaller or equal to M)");
        return NULL;
    }
    if ( (T->nt < A->nt) || (T->mt < A->mt) ) {
        dplasma_error("dplasma_zunglq_New", "illegal size of T (T should have as many tiles as A)");
        return NULL;
    }

    handle = parsec_zunglq_new( A, T, Q,
                               NULL );

    handle->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( handle->_g_p_work, ib * T->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( handle->arenas[PARSEC_zunglq_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* Upper triangular part of tile without diagonal */
    dplasma_add2arena_upper( handle->arenas[PARSEC_zunglq_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( handle->arenas[PARSEC_zunglq_LITTLE_T_ARENA],
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
 *  dplasma_zunglq_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zunglq_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunglq_New
 * @sa dplasma_zunglq
 *
 ******************************************************************************/
void
dplasma_zunglq_Destruct( parsec_handle_t *handle )
{
    parsec_zunglq_handle_t *parsec_zunglq = (parsec_zunglq_handle_t *)handle;

    parsec_matrix_del2arena( parsec_zunglq->arenas[PARSEC_zunglq_DEFAULT_ARENA   ] );
    parsec_matrix_del2arena( parsec_zunglq->arenas[PARSEC_zunglq_UPPER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zunglq->arenas[PARSEC_zunglq_LITTLE_T_ARENA  ] );

    parsec_private_memory_fini( parsec_zunglq->_g_p_work );
    free( parsec_zunglq->_g_p_work );

    parsec_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunglq - Generates of an M-by-N matrix Q with orthonormal rows,
 *  which is defined as the first M rows of a product of K elementary
 *  reflectors of order N
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgelqf_New().
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgelqf_New() routine.
 *          On entry, the i-th row must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgelqf_New() in the first k rows of its array
 *          argument A. M >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile LQ algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgelqf_New().
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
 * @sa dplasma_zunglq_New
 * @sa dplasma_zunglq_Destruct
 * @sa dplasma_cunglq
 * @sa dplasma_dorglq
 * @sa dplasma_sorglq
 * @sa dplasma_zgelqf
 *
 ******************************************************************************/
int
dplasma_zunglq( parsec_context_t *parsec,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *T,
                tiled_matrix_desc_t *Q )
{
    parsec_handle_t *parsec_zunglq;

    if (parsec == NULL) {
        dplasma_error("dplasma_zunglq", "dplasma not initialized");
        return -1;
    }
    if ( Q->m > Q->n) {
        dplasma_error("dplasma_zunglq", "illegal number of rows in Q (M)");
        return -2;
    }
    if ( A->m > Q->m) {
        dplasma_error("dplasma_zunglq", "illegal number of rows in A (K)");
        return -3;
    }
    if ( A->n != Q->n ) {
        dplasma_error("dplasma_zunglq", "illegal number of columns in A");
        return -5;
    }

    if (dplasma_imin(Q->m, dplasma_imin(Q->n, A->n)) == 0)
        return 0;

    parsec_zunglq = dplasma_zunglq_New(A, T, Q);

    if ( parsec_zunglq != NULL ){
        parsec_enqueue(parsec, parsec_zunglq);
        dplasma_progress(parsec);
        dplasma_zunglq_Destruct( parsec_zunglq );
    }
    return 0;
}
