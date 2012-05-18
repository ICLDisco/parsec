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
dplasma_zgetrs(dague_context_t *dague,
               const PLASMA_enum trans,
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

#ifdef DAGUE_COMPOSITION
    dague_object_t *dague_zlaswp = NULL;
    dague_object_t *dague_ztrsm1 = NULL;
    dague_object_t *dague_ztrsm2 = NULL;

    if ( trans == PlasmaNoTrans )
    {
        dague_zlaswp = dplasma_zlaswp_New(B, IPIV, 1);
        dague_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit, 1.0, A, B);
        dague_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, A, B);

        dague_enqueue( dague, dague_zlaswp );
        dague_enqueue( dague, dague_ztrsm1 );
        dague_enqueue( dague, dague_ztrsm2 );

        dplasma_progress( dague );

        dplasma_ztrsm_Destruct( dague_zlaswp );
        dplasma_ztrsm_Destruct( dague_ztrsm1 );
        dplasma_ztrsm_Destruct( dague_ztrsm2 );
    }
    else
    {
        dague_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, PlasmaUpper, trans, PlasmaNonUnit, 1.0, A, B);
        dague_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, PlasmaLower, trans, PlasmaUnit, 1.0, A, B);
        dague_zlaswp = dplasma_zlaswp_New(B, IPIV, -1);

        dague_enqueue( dague, dague_ztrsm1 );
        dague_enqueue( dague, dague_ztrsm2 );
        dague_enqueue( dague, dague_zlaswp );

        dplasma_progress( dague );

        dplasma_ztrsm_Destruct( dague_ztrsm1 );
        dplasma_ztrsm_Destruct( dague_ztrsm2 );
        dplasma_ztrsm_Destruct( dague_zlaswp );
    }
#else
    if ( trans == PlasmaNoTrans )
    {
        dplasma_zlaswp(dague, B, IPIV, 1);
        dplasma_ztrsm( dague, PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit,    1.0, A, B);
        dplasma_ztrsm( dague, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, A, B);
    }
    else
    {
        dplasma_ztrsm( dague, PlasmaLeft, PlasmaUpper, trans, PlasmaNonUnit, 1.0, A, B);
        dplasma_ztrsm( dague, PlasmaLeft, PlasmaLower, trans, PlasmaUnit,    1.0, A, B);
        dplasma_zlaswp(dague, B, IPIV, -1);
    }
#endif
    return 0;
}

