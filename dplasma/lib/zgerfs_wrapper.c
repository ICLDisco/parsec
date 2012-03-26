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
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zgerfs.h"



dague_object_t*
dplasma_zgerfs_New(tiled_matrix_desc_t *A, tiled_matrix_desc_t* LU,
                   tiled_matrix_desc_t* B, tiled_matrix_desc_t* R,
                   tiled_matrix_desc_t* X)
{
    dague_object_t *dague_zgerfs = NULL;

    dague_zgerfs = (dague_object_t*)dague_zgerfs_new(*A, (dague_ddesc_t*)A,
                                                     *LU, (dague_ddesc_t*)LU,
                                                     *B, (dague_ddesc_t*)B,
                                                     *R, (dague_ddesc_t*)R,
                                                     *X, (dague_ddesc_t*)X);

    dplasma_add2arena_tile(((dague_zgerfs_object_t*)dague_zgerfs)->arenas[DAGUE_zgerfs_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_zgerfs;
}

void
dplasma_zgerfs_Destruct( dague_object_t *o )
{
    dague_zgerfs_object_t *dague_zgerfs = (dague_zgerfs_object_t *)o;
    dplasma_datatype_undefine_type( &(dague_zgerfs->arenas[DAGUE_zgerfs_DEFAULT_ARENA   ]->opaque_dtt) );
    dague_zgerfs_destroy(dague_zgerfs);
}

int dplasma_zgerfs( dague_context_t *dague, tiled_matrix_desc_t* ddescA, tiled_matrix_desc_t* ddescLU,
                    tiled_matrix_desc_t* ddescB, tiled_matrix_desc_t* ddescR,
                    tiled_matrix_desc_t* ddescX)
{
    dague_object_t *dague_zgerfs = NULL;

    dague_zgerfs = dplasma_zgerfs_New(ddescA, ddescLU, ddescB, ddescR, ddescX);

    if ( dague_zgerfs != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zgerfs);
        dplasma_progress(dague);
        dplasma_zgerfs_Destruct( dague_zgerfs );
    }

    return 0;
}
