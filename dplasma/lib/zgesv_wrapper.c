/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "parsec/parsec_config.h"
#include "dplasma.h"
#include <core_blas.h>

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgesv - Solves a system of linear equations A * X = B with a
 * general square matrix A using the LU factorization with partial pivoting strategy
 * computed by dplasma_zgetrf().
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in,out] A
 *          Descriptor of the distributed factorized matrix A.
 *          On entry, describes the M-by-N matrix A.
 *          On exit, the factors L and U from the factorization
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 * @param[out] IPIV
 *          Descriptor of the IPIV matrix. Should be of size 1-by-min(M,N).
 *          On exit, contains the pivot indices; for 1 <= i <= min(M,N), row i
 *          of the matrix was interchanged with row IPIV(i).
 *
 * @param[in,out] B
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if return value = 0, B is overwritten by the solution matrix X.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgesv_New
 * @sa dplasma_zgesv_Destruct
 * @sa dplasma_cgesv
 * @sa dplasma_dgesv
 * @sa dplasma_sgesv
 *
 ******************************************************************************/
int
dplasma_zgesv( parsec_context_t *parsec,
               parsec_tiled_matrix_dc_t *A,
               parsec_tiled_matrix_dc_t *IPIV,
               parsec_tiled_matrix_dc_t *B)
{
    int info;

#ifdef PARSEC_COMPOSITION
#warning "Not implemented"

    parsec_taskpool_t *parsec_zgetrf = dplasma_zgetrf_New(A, IPIV, &info);
    parsec_taskpool_t *parsec_zlaswp = dplasma_zlaswp_New(B, IPIV, 1);
    parsec_taskpool_t *parsec_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit, 1.0, A, B);
    parsec_taskpool_t *parsec_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, A, B);

    parsec_enqueue( parsec, parsec_zgetrf  );
    parsec_enqueue( parsec, parsec_zlaswp );
    parsec_enqueue( parsec, parsec_ztrsm1 );
    parsec_enqueue( parsec, parsec_ztrsm2 );

    dplasma_wait_until_completion( parsec );

    dplasma_zgetrf_Destruct( parsec_zgetrf  );
    dplasma_zlaswp_Destruct( parsec_zlaswp );
    dplasma_ztrsm_Destruct( parsec_ztrsm1 );
    dplasma_ztrsm_Destruct( parsec_ztrsm2 );
#else
    info = dplasma_zgetrf(parsec, A, IPIV );
    if( info == 0 ) {
        dplasma_zgetrs( parsec, PlasmaNoTrans, A, IPIV, B );
    }
#endif

    return info;
}
