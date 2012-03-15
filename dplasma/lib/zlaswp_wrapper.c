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

#include "zlaswp.h"

dague_object_t *
dplasma_zlaswp_New(tiled_matrix_desc_t *A,
                   tiled_matrix_desc_t *IPIV,
                   int inc)
{
    dague_zlaswp_object_t *dague_laswp;

    dague_laswp = dague_zlaswp_new( *A,    (dague_ddesc_t*)A,
                                    *IPIV, (dague_ddesc_t*)IPIV,
                                    inc);

    /* A */
    dplasma_add2arena_tile( dague_laswp->arenas[DAGUE_zlaswp_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* IPIV */
    dplasma_add2arena_rectangle( dague_laswp->arenas[DAGUE_zlaswp_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    return (dague_object_t*)dague_laswp;
}

void
dplasma_zlaswp_Destruct( dague_object_t *o )
{
    dague_zlaswp_object_t *dague_zlaswp = (dague_zlaswp_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_zlaswp->arenas[DAGUE_zlaswp_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zlaswp->arenas[DAGUE_zlaswp_PIVOT_ARENA     ]->opaque_dtt) );

    dague_zlaswp_destroy(dague_zlaswp);
}

int
dplasma_zlaswp( dague_context_t *dague,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *IPIV,
                int inc)
{
    dague_object_t *dague_zlaswp = NULL;

    dague_zlaswp = dplasma_zlaswp_New(A, IPIV, inc);

    dague_enqueue( dague, (dague_object_t*)dague_zlaswp);
    dplasma_progress(dague);

    dplasma_zlaswp_Destruct( dague_zlaswp );

    return 0;
}
