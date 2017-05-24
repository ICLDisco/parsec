/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
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

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeqrs - Computes a minimum-norm solution min || A*X - B || using the
 * Hierarchical QR factorization A = Q*R computed by dplasma_zgeqrf_param().
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
 *          Descriptor of the matrix A of size M-by-N factorized with the
 *          dplasma_zgeqrf_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_New() in the first k columns of its array
 *          argument A.
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
 * @param[in,out] B
 *          Descriptor that covers both matrix B and X.
 *          On entry, the M-by-NRHS right hand side matrix B.
 *          On exit, the N-by-NRHS solution matrix X.
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
 * @sa dplasma_cgeqrs
 * @sa dplasma_dgeqrs
 * @sa dplasma_sgeqrs
 *
 ******************************************************************************/
int
dplasma_zgeqrs_param(parsec_context_t *parsec,
                     dplasma_qrtree_t *qrtree,
                     parsec_tiled_matrix_dc_t* A,
                     parsec_tiled_matrix_dc_t* TS,
                     parsec_tiled_matrix_dc_t* TT,
                     parsec_tiled_matrix_dc_t* B)
{
    parsec_tiled_matrix_dc_t *subA;
    parsec_tiled_matrix_dc_t *subB;

    /* Check input arguments */
    if ( A->n > A->m ) {
        dplasma_error("dplasma_zgeqrs", "illegal dimension of A, A->n > A->m");
        return -2;
    }
    if ( (TS->nt != A->nt) || (TS->mt != A->mt) ) {
        dplasma_error("dplasma_zgeqrs", "illegal size of TS (TS should have as many tiles as A)");
        return -3;
    }
    if ( (TT->nt != A->nt) || (TT->mt != A->mt) ) {
        dplasma_error("dplasma_zgeqrs", "illegal size of TT (TT should have as many tiles as A)");
        return -4;
    }
    if ( B->m < A->m ) {
        dplasma_error("dplasma_zgeqrs", "illegal dimension of B, (B->m < A->m)");
        return -5;
    }

    subA = tiled_matrix_submatrix( A, 0, 0, A->n, A->n );
    subB = tiled_matrix_submatrix( B, 0, 0, A->n, B->n );

    dplasma_zunmqr_param( parsec, PlasmaLeft, PlasmaConjTrans, qrtree, A, TS, TT, B );
    dplasma_ztrsm( parsec, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, subA, subB );

    free(subA);
    free(subB);

    return 0;
}

