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
#include "dague_internal.h"
#include "dplasma.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunglq_param_New - Generates the dague object that computes the generation
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
 *          \retval The dague object which describes the operation to perform
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
dague_handle_t*
dplasma_zunglq_param_New( dplasma_qrtree_t *qrtree,
                          tiled_matrix_desc_t *A,
                          tiled_matrix_desc_t *TS,
                          tiled_matrix_desc_t *TT,
                          tiled_matrix_desc_t *Q )
{
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

    return dplasma_zunmlq_param_New( PlasmaRight, PlasmaNoTrans, qrtree, A, TS, TT, Q );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zunglq_param_Destruct - Free the data structure associated to an object
 *  created with dplasma_zunglq_param_New().
 *
 *******************************************************************************
 *
 * @param[in,out] object
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunglq_param_New
 * @sa dplasma_zunglq_param
 *
 ******************************************************************************/
void
dplasma_zunglq_param_Destruct( dague_handle_t *object )
{
    dplasma_zunmlq_Destruct( object );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_Complex64_t
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
dplasma_zunglq_param( dague_context_t *dague,
                      dplasma_qrtree_t *qrtree,
                      tiled_matrix_desc_t *A,
                      tiled_matrix_desc_t *TS,
                      tiled_matrix_desc_t *TT,
                      tiled_matrix_desc_t *Q )
{
    if (dague == NULL) {
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

    return dplasma_zunmlq_param(dague, PlasmaRight, PlasmaNoTrans,
                                qrtree, A, TS, TT, Q);
}
