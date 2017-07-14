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
 * dplasma_zgelqs - Computes a minimum-norm solution min || A*X - B || using the
 * LQ factorization A = L*Q computed by dplasma_zgelqf().
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-N factorized with the
 *          dplasma_zgelqf_New() routine.
 *          On entry, the i-th row must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgelqf_New() in the first k rows of its array
 *          argument A.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile LQ algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgelqf_New().
 *
 * @param[in,out] B
 *          Descriptor that covers both matrix B and X.
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, the M-by-NRHS solution matrix X.
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
 * @sa dplasma_cgelqs
 * @sa dplasma_dgelqs
 * @sa dplasma_sgelqs
 *
 ******************************************************************************/
int
dplasma_zgelqs( parsec_context_t *parsec,
                parsec_tiled_matrix_dc_t* A,
                parsec_tiled_matrix_dc_t* T,
                parsec_tiled_matrix_dc_t* B )
{
    parsec_tiled_matrix_dc_t *subA;
    parsec_tiled_matrix_dc_t *subB;

    /* Check input arguments */
    if ( A->m > A->n ) {
        dplasma_error("dplasma_zgelqs", "illegal dimension of A, A->n > A->m");
        return -1;
    }
    if ( (T->nt != A->nt) || (T->mt != A->mt) ) {
        dplasma_error("dplasma_zgelqs", "illegal size of T (T should have as many tiles as A)");
        return -2;
    }
    if ( B->m < A->n ) {
        dplasma_error("dplasma_zgelqs", "illegal dimension of B, (B->m < A->n)");
        return -3;
    }

    subA = tiled_matrix_submatrix( A, 0, 0, A->m, A->m );
    subB = tiled_matrix_submatrix( B, 0, 0, A->m, B->n );

#ifdef PARSEC_COMPOSITION

    parsec_taskpool_t *parsec_zunmlq = NULL;
    parsec_taskpool_t *parsec_ztrsm  = NULL;

    parsec_ztrsm  = dplasma_ztrsm_New(  PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaNonUnit, 1.0, subA, subB );
    parsec_zunmlq = dplasma_zunmlq_New( PlasmaLeft, PlasmaConjTrans, A, T, B );

    parsec_enqueue( parsec, parsec_ztrsm );
    parsec_enqueue( parsec, parsec_zunmlq );

    dplasma_wait_until_completion( parsec );

    dplasma_ztrsm_Destruct( parsec_ztrsm );
    dplasma_ztrsm_Destruct( parsec_zunmlq );

#else

    dplasma_ztrsm(  parsec, PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaNonUnit, 1.0, subA, subB );
    dplasma_zunmlq( parsec, PlasmaLeft, PlasmaConjTrans, A, T, B );

#endif

    free(subA);
    free(subB);

    return 0;
}
