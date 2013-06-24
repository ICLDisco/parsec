/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"

int
dplasma_zgeqrs_param(dague_context_t *dague,
                     dplasma_qrtree_t *qrtree,
                     tiled_matrix_desc_t* A,
                     tiled_matrix_desc_t* TS,
                     tiled_matrix_desc_t* TT,
                     tiled_matrix_desc_t* B)
{

    dplasma_zunmqr_param( dague, PlasmaLeft, PlasmaConjTrans, qrtree, A, TS, TT, B );
    dplasma_ztrsm( dague, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, A, B );

    return 0;
}

