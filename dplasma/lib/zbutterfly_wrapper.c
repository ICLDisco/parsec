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
  
#include "zbutterfly.h"

dague_object_t* 
dplasma_zbutterfly_New( tiled_matrix_desc_t *A, int *info)
{
    dague_object_t *dague_zbutterfly = NULL;
    if( (A->nt)%2 || (A->mt)%2 ){
        dplasma_error("dplasma_zbutterfly_New", "illegal number of tiles in matrix");
        return NULL;
    }
    *info = 0;

    dague_zbutterfly = (dague_object_t *)dague_zbutterfly_new(*A, (dague_ddesc_t*)A);
    
    dplasma_add2arena_tile(((dague_zbutterfly_object_t*)dague_zbutterfly)->arenas[DAGUE_zbutterfly_DEFAULT_ARENA], 
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_zbutterfly;
}

void
dplasma_zbutterfly_Destruct( dague_object_t *o )
{
    dague_zbutterfly_object_t *obutterfly = (dague_zbutterfly_object_t *)o;
    
    dplasma_datatype_undefine_type( &(obutterfly->arenas[DAGUE_zbutterfly_DEFAULT_ARENA]->opaque_dtt) );

    dague_zbutterfly_destroy(obutterfly);
}
