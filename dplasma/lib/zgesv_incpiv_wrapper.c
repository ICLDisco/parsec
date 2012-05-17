/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
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
dplasma_zgesv_incpiv( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *L,
                      tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B)
{
    int info;

#ifdef DAGUE_COMPOSITION
    dague_object_t *dague_zgetrf  = dplasma_zgetrf_incpiv_New(A, L, IPIV, &info);
    dague_object_t *dague_ztrsmpl = dplasma_ztrsmpl_New(A, L, IPIV, B);
    dague_object_t *dague_ztrsm   = dplasma_ztrsm_New(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, A, B);

    dague_enqueue( dague, dague_zgetrf  );
    dague_enqueue( dague, dague_ztrsmpl );
    dague_enqueue( dague, dague_ztrsm   );

    dplasma_progress( dague );

    dplasma_zgetrf_incpiv_Destruct( dague_zgetrf  );
    dplasma_ztrsmpl_Destruct( dague_ztrsmpl );
    dplasma_ztrsm_Destruct( dague_ztrsm   );
#else
    info = dplasma_zgetrf_incpiv(dague, A, L, IPIV );
    if( info == 0 ) {
      dplasma_ztrsmpl(dague, A, L, IPIV, B );
      dplasma_ztrsm( dague, PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, A, B );
    }
#endif

    return info;
}
