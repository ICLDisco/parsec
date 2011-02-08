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
dplasma_zgesv( dague_context_t *dague, tiled_matrix_desc_t *A, tiled_matrix_desc_t *L, 
               tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B)
{
    int info;
    dague_object_t *dague_zgetrf  = dplasma_zgetrf_New(A, L, IPIV, &info);
    dague_object_t *dague_ztrsmpl = dplasma_ztrsmpl_New(A, L, IPIV, B);
    dague_object_t *dague_ztrsm   = dplasma_ztrsm_New(PlasmaLeft, PlasmaUpper, PlasmaNoTrans, PlasmaNonUnit, 1.0, A, B);
 
    dague_enqueue( dague, dague_zgetrf  );
    dague_enqueue( dague, dague_ztrsmpl );
    dague_enqueue( dague, dague_ztrsm   );

    dague_progress( dague );

    dplasma_ztrsm_Destruct( dague_zgetrf  );
    dplasma_ztrsm_Destruct( dague_ztrsmpl );
    dplasma_ztrsm_Destruct( dague_ztrsm   );

    return info;
}

