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
dplasma_zgesv( dague_context_t *dague,
               tiled_matrix_desc_t *A,
               tiled_matrix_desc_t *IPIV,
               tiled_matrix_desc_t *B)
{
    int info;

#ifdef DAGUE_COMPOSITION
#warning "Not implemented"

    dague_object_t *dague_zgetrf = dplasma_zgetrf_New(A, IPIV, &info);
    dague_object_t *dague_zlaswp = dplasma_zlaswp_New(B, IPIV, 1);
    dague_object_t *dague_ztrsm1 = dplasma_ztrsm_New(PlasmaLeft, PlasmaLower, PlasmaNoTrans, PlasmaUnit, 1.0, A, B);
    dague_object_t *dague_ztrsm2 = dplasma_ztrsm_New(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, A, B);

    dague_enqueue( dague, dague_zgetrf  );
    dague_enqueue( dague, dague_zlaswp );
    dague_enqueue( dague, dague_ztrsm1 );
    dague_enqueue( dague, dague_ztrsm2 );

    dplasma_progress( dague );

    dplasma_zgetrf_Destruct( dague_zgetrf  );
    dplasma_ztrsm_Destruct( dague_zlaswp );
    dplasma_ztrsm_Destruct( dague_ztrsm1 );
    dplasma_ztrsm_Destruct( dague_ztrsm2 );
#else
    info = dplasma_zgetrf(dague, A, IPIV );
    if( info == 0 ) {
        dplasma_zgetrs( dague, PlasmaNoTrans, A, IPIV, B );
    }
#endif

    return info;
}
