/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> c d s
 *
 */
#include "dague.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zlacpy.h"

/***************************************************************************//**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zlacpy_New - Generate a random matrix by tiles.
 *
 *******************************************************************************
 *
 * @param[out] A
 *          On exit, The random hermitian matrix A generated.
 *
 * @param[in] seed
 *          The seed used in the random generation.
 *
 ******************************************************************************/
dague_object_t* dplasma_zlacpy_New( PLASMA_enum uplo,
                                    tiled_matrix_desc_t *A,
                                    tiled_matrix_desc_t *B)
{
    dague_zlacpy_object_t* object;
    
    object = dague_zlacpy_new( uplo, *A, (dague_ddesc_t*)A, *B, (dague_ddesc_t*)B);

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zlacpy_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    return (dague_object_t*)object;
}

int dplasma_zlacpy( dague_context_t *dague, 
                    PLASMA_enum uplo,
                    tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *B) 
{
    dague_object_t *dague_zlacpy = NULL;

    dague_zlacpy = dplasma_zlacpy_New(uplo, A, B);

    dague_enqueue(dague, (dague_object_t*)dague_zlacpy);
    dplasma_progress(dague);

    dplasma_zlacpy_Destruct( dague_zlacpy );
    return 0;
}

void
dplasma_zlacpy_Destruct( dague_object_t *o )
{
    dague_zlacpy_object_t *dague_zlacpy = (dague_zlacpy_object_t *)o;
    dplasma_datatype_undefine_type( &(dague_zlacpy->arenas[DAGUE_zlacpy_DEFAULT_ARENA   ]->opaque_dtt) );
    dague_zlacpy_destroy(dague_zlacpy);
}

