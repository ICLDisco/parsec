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
dplasma_zhetrs(dague_context_t *dague, int uplo, const tiled_matrix_desc_t* A, tiled_matrix_desc_t* B)
{
    int info;
    dague_object_t *op;

    if( uplo != PlasmaLower ){
        dplasma_error("dplasma_zhetrs", "illegal value for \"uplo\".  Only PlasmaLower is currently supported");
    }

    // B = U_but_vec^T * B 
#if 1
    op = dplasma_zgebmm_New( B, 0, 0, 0, PlasmaConjTrans, &info);
    dague_enqueue(dague, op);
    dague_progress(dague);
    dplasma_zgebmm_Destruct(op);
    dplasma_zprint(dague, PlasmaUpperLower, B);
#endif

    dplasma_ztrsm( dague, PlasmaLeft, uplo, (uplo == PlasmaUpper) ? PlasmaConjTrans : PlasmaNoTrans, PlasmaUnit, 1.0, A, B );
    dplasma_ztrdsm( dague, A, B );
    dplasma_ztrsm( dague, PlasmaLeft, uplo, (uplo == PlasmaUpper) ? PlasmaNoTrans : PlasmaConjTrans, PlasmaUnit, 1.0, A, B );

    // X = U_but_vec * X  (here X is B)
#if 1
    op = dplasma_zgebmm_New( B, 0, 0, 0, PlasmaNoTrans, &info);
    dague_enqueue(dague, op);
    dague_progress(dague);
    dplasma_zgebmm_Destruct(op);
#endif

    return 0;
}

