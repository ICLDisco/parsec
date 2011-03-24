/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include <plasma.h>
#include <dague.h>
#include <scheduling.h>
#include "dplasma.h"

int
dplasma_zpotrs(dague_context_t *dague, const PLASMA_enum uplo, tiled_matrix_desc_t* A, tiled_matrix_desc_t* B)
{
    dague_object_t *dague_ztrsm1 = NULL;
    dague_object_t *dague_ztrsm2 = NULL;

    /* Check input arguments */
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_zpotrs", "illegal value of uplo");
        return -1;
    }

    if ( uplo == PlasmaUpper ) {
      dague_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0, A, B);
      dague_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaNoTrans,   PlasmaNonUnit, 1.0, A, B);
    } else {
      dague_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaNoTrans,   PlasmaNonUnit, 1.0, A, B);
      dague_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, uplo, PlasmaConjTrans, PlasmaNonUnit, 1.0, A, B);
    }

    dague_enqueue( dague, dague_ztrsm1 );
    dague_enqueue( dague, dague_ztrsm2 );

    dague_progress( dague );

    dplasma_ztrsm_Destruct( dague_ztrsm1 );
    dplasma_ztrsm_Destruct( dague_ztrsm2 );

    return 0;
}

