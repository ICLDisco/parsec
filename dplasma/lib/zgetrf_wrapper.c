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
#include "dplasma/lib/memory_pool.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "zgetrf.h"

void CORE_zgetrf_reclap_init(void);
void CORE_zgetrf_rectil_init(void);

/* Dirty hack which uses internal function of PLASMA to avoid initializing the lib */
/* TODO: remove check on context in PLASMA for this function, it's useless */
PLASMA_desc plasma_desc_init(PLASMA_enum dtyp, int mb, int nb, int bsiz,
                             int lm, int ln, int i, int j, int m, int n);


dague_object_t* dplasma_zgetrf_New(tiled_matrix_desc_t *A,
                                   tiled_matrix_desc_t *IPIV,
                                   int *INFO)
{
    dague_zgetrf_object_t *dague_getrf;
    
    if ( A->storage == matrix_Tile ) {
        CORE_zgetrf_rectil_init();
    } else {
        CORE_zgetrf_reclap_init();
    }

    dague_getrf = dague_zgetrf_new( *A, (dague_ddesc_t*)A,
                                    (dague_ddesc_t*)IPIV,
                                    INFO );

    /* A */
    dplasma_add2arena_tile( dague_getrf->arenas[DAGUE_zgetrf_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* IPIV */
    dplasma_add2arena_rectangle( dague_getrf->arenas[DAGUE_zgetrf_PIVOT_ARENA],
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    return (dague_object_t*)dague_getrf;
}

void
dplasma_zgetrf_Destruct( dague_object_t *o )
{
    dague_zgetrf_object_t *dague_zgetrf = (dague_zgetrf_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zgetrf->arenas[DAGUE_zgetrf_PIVOT_ARENA     ]->opaque_dtt) );

    dague_zgetrf_destroy(dague_zgetrf);
}

int dplasma_zgetrf( dague_context_t *dague,
                    tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *IPIV )
{
    dague_object_t *dague_zgetrf = NULL;

    int info = 0;
    dague_zgetrf = dplasma_zgetrf_New(A, IPIV, &info);

    dague_enqueue( dague, (dague_object_t*)dague_zgetrf);
    dplasma_progress(dague);

    return info;
}
