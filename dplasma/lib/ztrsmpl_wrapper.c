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
#include "dplasmaaux.h"
#include "dplasmatypes.h"

#include "generated/ztrsmpl.h"

dague_object_t *
dplasma_ztrsmpl_New(const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *L, 
                    const tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B)
{
    dague_ztrsmpl_object_t *dague_trsmpl = NULL; 

    dague_trsmpl = dague_ztrsmpl_new((dague_ddesc_t*)A, (dague_ddesc_t*)L, 
                                     (dague_ddesc_t*)IPIV, (dague_ddesc_t*)B, L->mb,
                                     A->m, A->n, A->mb, A->nb, A->mt, A->nt, L->mb, L->nb,
                                     B->m, B->n, B->mb, B->nb, B->mt, B->nt);
    
    /* A and B */
    dplasma_add2arena_tile( dague_trsmpl->arenas[DAGUE_ztrsmpl_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    /* IPIV */
    dplasma_add2arena_rectangle( dague_trsmpl->arenas[DAGUE_ztrsmpl_PIVOT_ARENA], 
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    /* L */
    dplasma_add2arena_rectangle( dague_trsmpl->arenas[DAGUE_ztrsmpl_SMALL_L_ARENA], 
                                 L->mb*L->nb*sizeof(Dague_Complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, L->mb, L->nb, -1);

    return (dague_object_t*)dague_trsmpl;
}

void
dplasma_ztrsmpl_Destruct( dague_object_t *o )
{
  dague_ztrsmpl_destroy((dague_ztrsmpl_object_t *)o);
}

void
dplasma_ztrsmpl( dague_context_t *dague, 
                 const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *L,
                 const tiled_matrix_desc_t *IPIV, tiled_matrix_desc_t *B)
{
    dague_object_t *dague_ztrsmpl = NULL;

    dague_ztrsmpl = dplasma_ztrsmpl_New(A, L, IPIV, B);

    if ( dague_ztrsmpl != NULL )
    {
        dague_enqueue( dague, dague_ztrsmpl );
        dague_progress( dague );
        
        dplasma_ztrsmpl_Destruct( dague_ztrsmpl );
    }
}
