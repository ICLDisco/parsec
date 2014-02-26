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
 * dplasma_zpotri - Computes the inverse of a complex Hermitian positive definite
 * matrix A using the Cholesky factorization A = U**H*U or A = L*L**H
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
 * @param[in,out] A
 *          Descriptor of the distributed factorized matrix A.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zpotri_New
 * @sa dplasma_zpotri_Destruct
 * @sa dplasma_cpotri
 * @sa dplasma_dpotri
 * @sa dplasma_spotri
 *
 ******************************************************************************/
int
dplasma_zpotri( dague_context_t *dague,
                PLASMA_enum uplo,
                tiled_matrix_desc_t* A )
{
    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_zpotri", "illegal value of uplo");
        return -1;
    }

#ifdef DAGUE_COMPOSITION
    dague_object_t *dague_ztrtri = NULL;
    dague_object_t *dague_zlauum = NULL;

    dague_ztrtri = dplasma_ztrtri_New(uplo, PlasmaNonUnit, A );
    dague_zlauum = dplasma_zlauum_New(uplo, A );

    dague_enqueue( dague, dague_ztrtri );
    dague_enqueue( dague, dague_zlauum );

    dplasma_progress( dague );

    dplasma_ztrtri_Destruct( dague_ztrtri );
    dplasma_zlauum_Destruct( dague_zlauum );
#else
    dplasma_ztrtri( dague, uplo, PlasmaNonUnit, A );
    dplasma_zlauum( dague, uplo, A );
#endif
    return 0;
}

