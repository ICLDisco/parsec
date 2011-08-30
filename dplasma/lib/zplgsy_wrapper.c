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

#include "zplgsy.h"

/***************************************************************************//**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zplgsy_New - Generate a random symmetric matrix by tiles.
 *
 *******************************************************************************
 *
 * @param[in] bump
 *          The value to add to the diagonal to be sure 
 *          to have a positive definite matrix.
 *
 * @param[in] uplo
 *          Specifies which elements of the matrix are to be set 
 *          = PlasmaUpper: Upper part of A is set;
 *          = PlasmaLower: Lower part of A is set;
 *          = PlasmaUpperLower: ALL elements of A are set.
 *
 * @param[out] A
 *          On exit, The random hermitian matrix A generated.
 *
 * @param[in] seed
 *          The seed used in the random generation.
 *
 ******************************************************************************/
dague_object_t* dplasma_zplgsy_New( Dague_Complex64_t bump, PLASMA_enum uplo, 
                                    tiled_matrix_desc_t *A,
                                    unsigned long long int seed)
{
    dague_zplgsy_object_t* object;
    
    object = dague_zplgsy_new( uplo, bump, seed, *A, (dague_ddesc_t*)A);

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zplgsy_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    return (dague_object_t*)object;
}

int dplasma_zplgsy( dague_context_t *dague, 
                    Dague_Complex64_t bump, PLASMA_enum uplo, 
                    tiled_matrix_desc_t *A,
                    unsigned long long int seed) 
{
    dague_object_t *dague_zplgsy = NULL;

    dague_zplgsy = dplasma_zplgsy_New(bump, uplo, A, seed);

    dague_enqueue(dague, (dague_object_t*)dague_zplgsy);
    dplasma_progress(dague);

    dplasma_zplgsy_Destruct( dague_zplgsy );
    return 0;
}

void
dplasma_zplgsy_Destruct( dague_object_t *o )
{
    dague_zplgsy_object_t *dague_zplgsy = (dague_zplgsy_object_t *)o;
    dague_zplgsy_destroy(dague_zplgsy);
}

