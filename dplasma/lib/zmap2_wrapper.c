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
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map2.h"

dague_object_t *
dplasma_zmap2_New( PLASMA_enum uplo,
                   tiled_matrix_desc_t *A, 
                   tiled_matrix_desc_t *B,
                   dague_operator_t operator, 
                   void * op_args)
{
    dague_map2_object_t *dague_map2 = NULL; 
    
    dague_map2 = dague_map2_new( uplo, 
                                 *A, (dague_ddesc_t*)A,
                                 *B, (dague_ddesc_t*)B,
                                 operator, op_args);
    
    dplasma_add2arena_tile( dague_map2->arenas[DAGUE_map2_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb);
    
    return (dague_object_t*)dague_map2;
}

void
dplasma_zmap2_Destruct( dague_object_t *o )
{
    dague_map2_object_t *omap2 = (dague_map2_object_t *)o;

    dplasma_datatype_undefine_type( &(omap2->arenas[DAGUE_map2_DEFAULT_ARENA]->opaque_dtt) );
    
    dague_map2_destroy( omap2 );
}

void
dplasma_zmap2( dague_context_t *dague,
               PLASMA_enum uplo,
               tiled_matrix_desc_t *A, 
               tiled_matrix_desc_t *B,
               dague_operator_t operator, 
               void * op_args)
{
    dague_object_t *dague_map2 = NULL;

    dague_map2 = dplasma_zmap2_New( uplo, A, B, operator, op_args );

    if ( dague_map2 != NULL ) 
    {
        dague_enqueue( dague, dague_map2 );
        dplasma_progress( dague );
        dplasma_zmap2_Destruct( dague_map2 );
    }
}
