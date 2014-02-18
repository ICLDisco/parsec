/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
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
#include "dplasma/lib/dplasmatypes.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zpotrs - Solves a system of linear equations A * X = B with a
 * symmetric positive definite (or Hermitian positive definite in the complex
 * case) matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
 * computed by dplasma_zpotrf().
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of A is referenced;
 *          = PlasmaLower: Lower triangle of A is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed factorized matrix A.
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
 * @sa dplasma_zpotrs_New
 * @sa dplasma_zpotrs_Destruct
 * @sa dplasma_cpotrs
 * @sa dplasma_dpotrs
 * @sa dplasma_spotrs
 *
 ******************************************************************************/
int
dplasma_zpotrs( dague_context_t *dague,
                PLASMA_enum uplo,
                const tiled_matrix_desc_t* A,
                tiled_matrix_desc_t* B )
{
    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_zpotrs", "illegal value of uplo");
        return -1;
    }

#ifdef DAGUE_COMPOSITION
    dague_handle_t *dague_ztrsm1 = NULL;
    dague_handle_t *dague_ztrsm2 = NULL;

    if ( uplo == PlasmaUpper ) {
      dague_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0, A, B);
      dague_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaNoTrans,   PlasmaNonUnit, 1.0, A, B);
    } else {
      dague_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaNoTrans,   PlasmaNonUnit, 1.0, A, B);
      dague_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0, A, B);
    }

    dague_enqueue( dague, dague_ztrsm1 );
    dague_enqueue( dague, dague_ztrsm2 );

    dplasma_progress( dague );

    dplasma_ztrsm_Destruct( dague_ztrsm1 );
    dplasma_ztrsm_Destruct( dague_ztrsm2 );
#else
    if ( uplo == PlasmaUpper ) {
      dplasma_ztrsm( dague, PlasmaLeft, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0, A, B );
      dplasma_ztrsm( dague, PlasmaLeft, uplo, PlasmaNoTrans,   PlasmaNonUnit, 1.0, A, B );
    } else {
      dplasma_ztrsm( dague, PlasmaLeft, uplo, PlasmaNoTrans,   PlasmaNonUnit, 1.0, A, B );
      dplasma_ztrsm( dague, PlasmaLeft, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0, A, B );
    }
#endif
    return 0;
}

