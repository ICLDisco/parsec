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

#include "zaxpy.h"

/***************************************************************************//**
 *
 * @ingroup Dague_Complex64_t
 *
 *  dplasma_zaxpy_New - Compute the operation B = alpha * A + B
 *
 *******************************************************************************
 *
 * @param[in] alpha
 *          The scalar alpha
 *
 * @param[in] A
 *          The matrix A of size M-by-N
 *
 * @param[in,out] B
 *          On entry, the matrix B of size equal or greater to M-by-N
 *          On exit, the matrix B with the M-by-N part overwrite by alpha*A+B
 *
 ******************************************************************************/
dague_object_t* dplasma_zaxpy_New( Dague_Complex64_t alpha,
				   tiled_matrix_desc_t *A,
				   tiled_matrix_desc_t *B)
{
    dague_zaxpy_object_t* object;
    
    object = dague_zaxpy_new( alpha, *A, (dague_ddesc_t*)A, *B, (dague_ddesc_t*)B);

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zaxpy_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    return (dague_object_t*)object;
}

int dplasma_zaxpy( dague_context_t *dague, 
		   Dague_Complex64_t alpha,
		   tiled_matrix_desc_t *A,
		   tiled_matrix_desc_t *B) 
{
    dague_object_t *dague_zaxpy = NULL;

    dague_zaxpy = dplasma_zaxpy_New(alpha, A, B);

    dague_enqueue(dague, (dague_object_t*)dague_zaxpy);
    dplasma_progress(dague);

    dplasma_zaxpy_Destruct( dague_zaxpy );
    return 0;
}

void
dplasma_zaxpy_Destruct( dague_object_t *o )
{
    dague_zaxpy_object_t *dague_zaxpy = (dague_zaxpy_object_t *)o;
    dplasma_datatype_undefine_type( &(dague_zaxpy->arenas[DAGUE_zaxpy_DEFAULT_ARENA   ]->opaque_dtt) );
    dague_zaxpy_destroy(dague_zaxpy);
}

