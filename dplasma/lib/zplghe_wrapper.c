/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"

#include "zplghe.h"

/***************************************************************************//**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zplghe_New - Generate a random hermitian matrix by tiles.
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
dague_object_t* dplasma_zplghe_New( double bump, PLASMA_enum uplo, 
                                    tiled_matrix_desc_t *A,
                                    unsigned long long int seed)
{
    dague_zplghe_object_t* object;
    
    object = dague_zplghe_new( uplo, bump, seed, *A, (dague_ddesc_t*)A);

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zplghe_DEFAULT_ARENA], 
                            A->mb*A->nb*sizeof(Dague_Complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );
    
    return (dague_object_t*)object;
}

int dplasma_zplghe( dague_context_t *dague, 
                    double bump, PLASMA_enum uplo, 
                    tiled_matrix_desc_t *A,
                    unsigned long long int seed) 
{
    dague_object_t *dague_zplghe = NULL;

    dague_zplghe = dplasma_zplghe_New(bump, uplo, A, seed);

    dague_enqueue(dague, (dague_object_t*)dague_zplghe);
    dplasma_progress(dague);

    dplasma_zplghe_Destruct( dague_zplghe );
    return 0;
}

void
dplasma_zplghe_Destruct( dague_object_t *o )
{
    dague_zplghe_object_t *dague_zplghe = (dague_zplghe_object_t *)o;
    dplasma_datatype_undefine_type( &(dague_zplghe->arenas[DAGUE_zplghe_DEFAULT_ARENA]->opaque_dtt) );
    dague_zplghe_destroy(dague_zplghe);
}

