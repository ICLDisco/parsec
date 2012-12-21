/*
 * Copyright (c) 2011-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> c d s
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zprint.h"

/***************************************************************************//**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zprint_New - Generate a random matrix by tiles.
 *
 *******************************************************************************
 *
 * @param[out] A
 *          On exit, The random hermitian matrix A generated.
 *
 ******************************************************************************/
int dplasma_zprint( dague_context_t *dague, 
                    PLASMA_enum uplo,
                    tiled_matrix_desc_t *A) 
{
    dague_zprint_handle_t* handle;
    
    handle = dague_zprint_new( uplo, *A, (dague_ddesc_t*)A);

    /* Default type */
    dplasma_add2arena_tile( handle->arenas[DAGUE_zprint_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    dague_enqueue(dague, (dague_handle_t*)handle);
    dplasma_progress(dague);

    dplasma_datatype_undefine_type( &(handle->arenas[DAGUE_zprint_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_HANDLE_DESTRUCT( handle );

    return 0;
}
