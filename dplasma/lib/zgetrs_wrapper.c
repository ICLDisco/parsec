/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include <core_blas.h>
#include "dplasma.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgetrs - Solves a system of linear equations A * X = B with a
 * general square matrix A using the LU factorization with partial pivoting strategy
 * computed by dplasma_zgetrf().
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is transposed, not transposed or
 *          conjugate transposed:
 *          = PlasmaNoTrans:   A is transposed;
 *          = PlasmaTrans:     A is not transposed;
 *          = PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] A
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
 * @sa dplasma_zgetrs_New
 * @sa dplasma_zgetrs_Destruct
 * @sa dplasma_cgetrs
 * @sa dplasma_dgetrs
 * @sa dplasma_sgetrs
 *
 ******************************************************************************/
int
dplasma_zgetrs(parsec_context_t *parsec,
               PLASMA_enum trans,
               tiled_matrix_desc_t *A,
               tiled_matrix_desc_t *IPIV,
               tiled_matrix_desc_t *B)
{
    /* Check input arguments */
    if ( trans != PlasmaNoTrans &&
         trans != PlasmaTrans   &&
         trans != PlasmaConjTrans ) {
        dplasma_error("dplasma_zgetrs", "illegal value of trans");
        return -1;
    }

#ifdef PARSEC_COMPOSITION
    parsec_handle_t *parsec_zlaswp = NULL;
    parsec_handle_t *parsec_ztrsm1 = NULL;
    parsec_handle_t *parsec_ztrsm2 = NULL;

    if ( trans == PlasmaNoTrans )
    {
        parsec_zlaswp = dplasma_zlaswp_New(B, IPIV, 1);
        parsec_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit, 1.0, A, B);
        parsec_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, A, B);

        parsec_enqueue( parsec, parsec_zlaswp );
        parsec_enqueue( parsec, parsec_ztrsm1 );
        parsec_enqueue( parsec, parsec_ztrsm2 );

        dplasma_progress( parsec );

        dplasma_ztrsm_Destruct( parsec_zlaswp );
        dplasma_ztrsm_Destruct( parsec_ztrsm1 );
        dplasma_ztrsm_Destruct( parsec_ztrsm2 );
    }
    else
    {
        parsec_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, PlasmaUpper, trans, PlasmaNonUnit, 1.0, A, B);
        parsec_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, PlasmaLower, trans, PlasmaUnit, 1.0, A, B);
        parsec_zlaswp = dplasma_zlaswp_New(B, IPIV, -1);

        parsec_enqueue( parsec, parsec_ztrsm1 );
        parsec_enqueue( parsec, parsec_ztrsm2 );
        parsec_enqueue( parsec, parsec_zlaswp );

        dplasma_progress( parsec );

        dplasma_ztrsm_Destruct( parsec_ztrsm1 );
        dplasma_ztrsm_Destruct( parsec_ztrsm2 );
        dplasma_ztrsm_Destruct( parsec_zlaswp );
    }
#else
    if ( trans == PlasmaNoTrans )
    {
        dplasma_zlaswp(parsec, B, IPIV, 1);
        dplasma_ztrsm( parsec, PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit,    1.0, A, B);
        dplasma_ztrsm( parsec, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, A, B);
    }
    else
    {
        dplasma_ztrsm( parsec, PlasmaLeft, PlasmaUpper, trans, PlasmaNonUnit, 1.0, A, B);
        dplasma_ztrsm( parsec, PlasmaLeft, PlasmaLower, trans, PlasmaUnit,    1.0, A, B);
        dplasma_zlaswp(parsec, B, IPIV, -1);
    }
#endif
    return 0;
}

