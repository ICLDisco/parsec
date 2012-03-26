/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "dplasma.h"

int
dplasma_zhetrs(dague_context_t *dague, const tiled_matrix_desc_t* A, tiled_matrix_desc_t* B)
{

    dplasma_ztrsm( dague, PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit, 1.0, A, B );
    dplasma_ztrdsm( dague, A, B );
    dplasma_ztrsm( dague, PlasmaLeft, PlasmaLower, PlasmaConjTrans, PlasmaUnit, 1.0, A, B );

    return 0;
}

