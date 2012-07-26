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
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/dplasmatypes.h"

#include "ztrsmpl.h"
#include "ztrsmpl_sd.h"

dague_object_t *
dplasma_ztrsmpl_New(const tiled_matrix_desc_t *A, 
                    const tiled_matrix_desc_t *L, 
                    const tiled_matrix_desc_t *IPIV, 
                    tiled_matrix_desc_t *B)
{
    dague_ztrsmpl_object_t *dague_trsmpl = NULL; 

    dague_trsmpl = dague_ztrsmpl_new(*A, (dague_ddesc_t*)A, 
                                     *L, (dague_ddesc_t*)L, 
                                         (dague_ddesc_t*)IPIV, 
                                     *B, (dague_ddesc_t*)B, 
                                     L->mb);

    /* A and B */
    dplasma_add2arena_tile( dague_trsmpl->arenas[DAGUE_ztrsmpl_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    /* IPIV */
    dplasma_add2arena_rectangle( dague_trsmpl->arenas[DAGUE_ztrsmpl_PIVOT_ARENA], 
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    /* L */
    dplasma_add2arena_rectangle( dague_trsmpl->arenas[DAGUE_ztrsmpl_SMALL_L_ARENA], 
                                 L->mb*L->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, L->mb, L->nb, -1);

    return (dague_object_t*)dague_trsmpl;
}

void
dplasma_ztrsmpl_Destruct( dague_object_t *o )
{
    dague_ztrsmpl_object_t *dague_trsmpl = (dague_ztrsmpl_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_trsmpl->arenas[DAGUE_ztrsmpl_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_trsmpl->arenas[DAGUE_ztrsmpl_PIVOT_ARENA  ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_trsmpl->arenas[DAGUE_ztrsmpl_SMALL_L_ARENA]->opaque_dtt) );
      
    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
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
        dplasma_progress( dague );
        
        dplasma_ztrsmpl_Destruct( dague_ztrsmpl );
    }
}

/****************************************************************/
/*
 * Single data version grouping L and IPIV in L
 */
dague_object_t *
dplasma_ztrsmpl_sd_New(const tiled_matrix_desc_t *A, 
                       const tiled_matrix_desc_t *L, 
                       tiled_matrix_desc_t *B)
{
    dague_ztrsmpl_sd_object_t *dague_trsmpl_sd = NULL; 

    dague_trsmpl_sd = dague_ztrsmpl_sd_new(*A, (dague_ddesc_t*)A, 
                                           *L, (dague_ddesc_t*)L, 
                                           *B, (dague_ddesc_t*)B, 
                                           L->mb-1);
    
    /* A and B */
    dplasma_add2arena_tile( dague_trsmpl_sd->arenas[DAGUE_ztrsmpl_sd_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    /* IPIV */
    dplasma_add2arena_rectangle( dague_trsmpl_sd->arenas[DAGUE_ztrsmpl_sd_PIVOT_ARENA], 
                                 A->mb*sizeof(int),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_INT, A->mb, 1, -1 );

    /* L */
    dplasma_add2arena_rectangle( dague_trsmpl_sd->arenas[DAGUE_ztrsmpl_sd_L_PIVOT_ARENA], 
                                 L->mb*L->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, L->mb, L->nb, -1);

    return (dague_object_t*)dague_trsmpl_sd;
}

void
dplasma_ztrsmpl_sd_Destruct( dague_object_t *o )
{
    dague_ztrsmpl_sd_object_t *dague_ztrsmpl_sd = (dague_ztrsmpl_sd_object_t *)o;

    dplasma_datatype_undefine_type( &(dague_ztrsmpl_sd->arenas[DAGUE_ztrsmpl_sd_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_ztrsmpl_sd->arenas[DAGUE_ztrsmpl_sd_PIVOT_ARENA  ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_ztrsmpl_sd->arenas[DAGUE_ztrsmpl_sd_L_PIVOT_ARENA]->opaque_dtt) );
      
    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

void
dplasma_ztrsmpl_sd( dague_context_t *dague, 
                    const tiled_matrix_desc_t *A, 
                    const tiled_matrix_desc_t *L,
                    tiled_matrix_desc_t *B)
{
    dague_object_t *dague_ztrsmpl_sd = NULL;

    dague_ztrsmpl_sd = dplasma_ztrsmpl_sd_New(A, L, B);

    if ( dague_ztrsmpl_sd != NULL )
    {
        dague_enqueue( dague, dague_ztrsmpl_sd );
        dplasma_progress( dague );
        
        dplasma_ztrsmpl_sd_Destruct( dague_ztrsmpl_sd );
    }
}

