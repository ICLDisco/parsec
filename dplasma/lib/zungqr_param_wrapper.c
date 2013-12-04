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
 *  dplasma_zungqr_param_New - Generates the dague object that computes the generation
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
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgeqrf_param_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_param_New() in the first k columns of its array
 *          argument A. N >= K >= 0.
 *
 * @param[out] TS
 *          Descriptor of the matrix TS distributed exactly as the A matrix. TS.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb == A.nb.
 *          On exit, contains auxiliary information computed through TS kernels
 *          at the lowest level and which are required to generate the Q matrix,
 *          and/or solve the problem.
 *
 * @param[out] TT
 *          Descriptor of the matrix TT distributed exactly as the A matrix. TT.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb == A.nb.
 *          On exit, contains auxiliary information computed through TT kernels
 *          at higher levels and which are required to generate the Q matrix,
 *          and/or solve the problem.
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
 *          \retval The dague object which describes the operation to perform
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
dague_object_t*
dplasma_zungqr_param_New( dplasma_qrtree_t *qrtree,
                          tiled_matrix_desc_t *A,
                          tiled_matrix_desc_t *TS,
                          tiled_matrix_desc_t *TT,
                          tiled_matrix_desc_t *Q)
{
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

    return dplasma_zunmqr_param_New( PlasmaLeft, PlasmaNoTrans, qrtree, A, TS, TT, Q );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zungqr_param_Destruct - Free the data structure associated to an object
 *  created with dplasma_zungqr_param_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zungqr_param_New
 * @sa dplasma_zungqr_param
 *
 ******************************************************************************/
void
dplasma_zungqr_param_Destruct( dague_object_t *o )
{
    dplasma_zunmqr_param_Destruct( o );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zungqr_param - Computes the generation of an M-by-N matrix Q with
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
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgeqrf_param_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_param_New() in the first k columns of its array
 *          argument A. N >= K >= 0.
 *
 * @param[out] TS
 *          Descriptor of the matrix TS distributed exactly as the A matrix. TS.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb == A.nb.
 *          On exit, contains auxiliary information computed through TS kernels
 *          at the lowest level and which are required to generate the Q matrix,
 *          and/or solve the problem.
 *
 * @param[out] TT
 *          Descriptor of the matrix TT distributed exactly as the A matrix. TT.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb == A.nb.
 *          On exit, contains auxiliary information computed through TT kernels
 *          at higher levels and which are required to generate the Q matrix,
 *          and/or solve the problem.
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
                      tiled_matrix_desc_t *Q)
{
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

    return dplasma_zunmqr_param(dague, PlasmaLeft, PlasmaNoTrans, qrtree, A, TS, TT, Q);
}
