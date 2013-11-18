/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <core_blas.h>
#include "dplasma.h"

int
dplasma_zgeqrs(dague_context_t *dague,
               tiled_matrix_desc_t* A,
               tiled_matrix_desc_t* T,
               tiled_matrix_desc_t* B)
{

    dplasma_zunmqr( dague, PlasmaLeft, PlasmaConjTrans, A, T, B );
    dplasma_ztrsm( dague, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, A, B );

    return 0;
}


