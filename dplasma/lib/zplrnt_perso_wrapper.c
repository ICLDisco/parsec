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

#include "zplrnt_perso.h"

/***************************************************************************//**
 *
 * @ingroup DPLASMA_Complex64_t
 *
 *  dplasma_zplrnt_perso_New - Generate a random matrix by tiles.
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
dague_object_t* dplasma_zplrnt_perso_New( tiled_matrix_desc_t *A,
                                          enum matrix_init_e type,
                                          unsigned long long int seed )
{
    dague_zplrnt_perso_object_t* object;

    object = dague_zplrnt_perso_new( type, seed, (dague_ddesc_t*)A );

    /* Default type */
    dplasma_add2arena_tile( object->arenas[DAGUE_zplrnt_perso_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    return (dague_object_t*)object;
}

int dplasma_zplrnt_perso( dague_context_t *dague,
                          tiled_matrix_desc_t *A,
                          enum matrix_init_e type,
                          unsigned long long int seed )
{
    dague_object_t *dague_zplrnt_perso = NULL;

    dague_zplrnt_perso = dplasma_zplrnt_perso_New( A, type, seed );

    dague_enqueue(dague, (dague_object_t*)dague_zplrnt_perso);
    dplasma_progress(dague);

    dplasma_zplrnt_perso_Destruct( dague_zplrnt_perso );
    return 0;
}

void
dplasma_zplrnt_perso_Destruct( dague_object_t *o )
{
    dague_zplrnt_perso_object_t *dague_zplrnt_perso = (dague_zplrnt_perso_object_t *)o;
    dplasma_datatype_undefine_type( &(dague_zplrnt_perso->arenas[DAGUE_zplrnt_perso_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(dague_zplrnt_perso);
}
