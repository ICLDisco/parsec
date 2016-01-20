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
#include "dague/private_mempool.h"

#include "zungqr_param.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zungqr_param_New - Generates the dague handle that computes the generation
 *  of an M-by-N matrix Q with orthonormal columns, which is defined as the
 *  first N columns of a product of K elementary reflectors of order M
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgeqrf_param_New().
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
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgeqrf_param_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_param_New() in the first k columns of its array
 *          argument A. N >= K >= 0.
 *
 * @param[in] TS
 *          Descriptor of the matrix TS distributed exactly as the A
 *          matrix. TS.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
 *
 * @param[in] TT
 *          Descriptor of the matrix TT distributed exactly as the A
 *          matrix. TT.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
 *
 * @param[in,out] Q
 *          Descriptor of the M-by-N matrix Q with orthonormal columns.
 *          On entry, the Id matrix.
 *          On exit, the orthonormal matrix Q.
 *          M >= N >= 0.
 *
 *******************************************************************************
 *
 * @return
 *          \retval The dague handle which describes the operation to perform
 *                  NULL if one of the parameter is incorrect
 *
 *******************************************************************************
 *
 * @sa dplasma_zungqr_param_Destruct
 * @sa dplasma_zungqr_param
 * @sa dplasma_cungqr_param_New
 * @sa dplasma_dorgqr_param_New
 * @sa dplasma_sorgqr_param_New
 * @sa dplasma_zgeqrf_param_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zungqr_param_New( dplasma_qrtree_t *qrtree,
                          tiled_matrix_desc_t *A,
                          tiled_matrix_desc_t *TS,
                          tiled_matrix_desc_t *TT,
                          tiled_matrix_desc_t *Q )
{
    dague_zungqr_param_handle_t* handle;
    int ib = TS->mb;

    if ( Q->n > Q->m ) {
        dplasma_error("dplasma_zungqr_param_New", "illegal size of Q (N should be smaller or equal to M)");
        return NULL;
    }
    if ( A->n > Q->n ) {
        dplasma_error("dplasma_zungqr_param_New", "illegal size of A (K should be smaller or equal to N)");
        return NULL;
    }
    if ( (TS->nt < A->nt) || (TS->mt < A->mt) ) {
        dplasma_error("dplasma_zungqr_param_New", "illegal size of TS (TS should have as many tiles as A)");
        return NULL;
    }
    if ( (TT->nt < A->nt) || (TT->mt < A->mt) ) {
        dplasma_error("dplasma_zungqr_param_New", "illegal size of TT (TT should have as many tiles as A)");
        return NULL;
    }

    handle = dague_zungqr_param_new( (dague_ddesc_t*)A,
                                     (dague_ddesc_t*)TS,
                                     (dague_ddesc_t*)TT,
                                     (dague_ddesc_t*)Q,
                                     *qrtree,
                                     NULL );

    handle->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( handle->p_work, ib * TS->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( handle->arenas[DAGUE_zungqr_param_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            dague_datatype_double_complex_t, A->mb );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( handle->arenas[DAGUE_zungqr_param_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             dague_datatype_double_complex_t, A->mb, 1 );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( handle->arenas[DAGUE_zungqr_param_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             dague_datatype_double_complex_t, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( handle->arenas[DAGUE_zungqr_param_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_double_complex_t, TS->mb, TS->nb, -1);

    return (dague_handle_t*)handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zungqr_param_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zungqr_param_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zungqr_param_New
 * @sa dplasma_zungqr_param
 *
 ******************************************************************************/
void
dplasma_zungqr_param_Destruct( dague_handle_t *handle )
{
    dague_zungqr_param_handle_t *dague_zungqr = (dague_zungqr_param_handle_t *)handle;

    dague_matrix_del2arena( dague_zungqr->arenas[DAGUE_zungqr_param_DEFAULT_ARENA   ] );
    dague_matrix_del2arena( dague_zungqr->arenas[DAGUE_zungqr_param_UPPER_TILE_ARENA] );
    dague_matrix_del2arena( dague_zungqr->arenas[DAGUE_zungqr_param_LOWER_TILE_ARENA] );
    dague_matrix_del2arena( dague_zungqr->arenas[DAGUE_zungqr_param_LITTLE_T_ARENA  ] );

    dague_private_memory_fini( dague_zungqr->p_work );
    free( dague_zungqr->p_work );

    handle->destructor(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zungqr_param - Generates an M-by-N matrix Q with
 *  orthonormal columns, which is defined as the first N columns of a product of
 *  K elementary reflectors of order M
 *
 *     Q  =  H(1) H(2) . . . H(k)
 *
 * as returned by dplasma_zgeqrf_param_New().
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] qrtree
 *          The structure that describes the trees used to perform the
 *          hierarchical QR factorization.
 *          See dplasma_hqr_init() or dplasma_systolic_init().
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgeqrf_param_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_param_New() in the first k columns of its array
 *          argument A. N >= K >= 0.
 *
 * @param[in] TS
 *          Descriptor of the matrix TS distributed exactly as the A
 *          matrix. TS.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
 *
 * @param[in] TT
 *          Descriptor of the matrix TT distributed exactly as the A
 *          matrix. TT.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
 *
 * @param[in,out] Q
 *          Descriptor of the M-by-N matrix Q with orthonormal columns.
 *          On entry, the Id matrix.
 *          On exit, the orthonormal matrix Q.
 *          M >= N >= 0.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zungqr_param_New
 * @sa dplasma_zungqr_param_Destruct
 * @sa dplasma_cungqr_param
 * @sa dplasma_dorgqr_param
 * @sa dplasma_sorgqr_param
 * @sa dplasma_zgeqrf_param
 *
 ******************************************************************************/
int
dplasma_zungqr_param( dague_context_t *dague,
                      dplasma_qrtree_t *qrtree,
                      tiled_matrix_desc_t *A,
                      tiled_matrix_desc_t *TS,
                      tiled_matrix_desc_t *TT,
                      tiled_matrix_desc_t *Q )
{
    dague_handle_t *dague_zungqr;

    if (dague == NULL) {
        dplasma_error("dplasma_zungqr_param", "dplasma not initialized");
        return -1;
    }
    if ( Q->n > Q->m) {
        dplasma_error("dplasma_zungqr_param", "illegal number of columns in Q (N)");
        return -2;
    }
    if ( A->n > Q->n) {
        dplasma_error("dplasma_zungqr_param", "illegal number of columns in A (K)");
        return -3;
    }
    if ( A->m != Q->m ) {
        dplasma_error("dplasma_zungqr_param", "illegal number of rows in A");
        return -5;
    }

    if (dplasma_imin(Q->m, dplasma_imin(Q->n, A->n)) == 0)
        return 0;

    dplasma_qrtree_check( A, qrtree );

    dague_zungqr = dplasma_zungqr_param_New(qrtree, A, TS, TT, Q);

    if ( dague_zungqr != NULL ){
        dague_enqueue(dague, dague_zungqr);
        dplasma_progress(dague);
        dplasma_zungqr_param_Destruct( dague_zungqr );
    }
    return 0;
}
