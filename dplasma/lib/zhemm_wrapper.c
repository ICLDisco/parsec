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

#include "zhemm.h"

dague_object_t*
dplasma_zhemm_New( const PLASMA_enum side, const PLASMA_enum uplo,
                   const Dague_Complex64_t alpha, const tiled_matrix_desc_t* A, const tiled_matrix_desc_t* B,
                   const double beta,  tiled_matrix_desc_t* C)
{
    dague_zhemm_object_t* object;

    /* Check input arguments */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("PLASMA_zhemm", "illegal value of side");
        return NULL /*-1*/;
    }
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("PLASMA_zhemm", "illegal value of uplo");
        return NULL /*-2*/;
    }

    object = dague_zhemm_new(side, uplo, alpha, beta, 
                             *A, (dague_ddesc_t*)A, 
                             *B, (dague_ddesc_t*)B, 
                             *C, (dague_ddesc_t*)C);

    dplasma_add2arena_tile(object->arenas[DAGUE_zhemm_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, C->mb);

    return (dague_object_t*)object;
}

void
dplasma_zhemm_Destruct( dague_object_t *o )
{
    dague_zhemm_object_t *zhemm_object = (dague_zhemm_object_t*)o;
    dplasma_datatype_undefine_type( &(zhemm_object->arenas[DAGUE_zhemm_DEFAULT_ARENA]->opaque_dtt) );
    dague_zhemm_destroy(zhemm_object);
}

void
dplasma_zhemm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo,
               const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
               const double beta,  tiled_matrix_desc_t *C)
{
    dague_object_t *dague_zhemm = NULL;

    dague_zhemm = dplasma_zhemm_New(side, uplo, 
                                    alpha, A, B,
                                    beta, C);

    if ( dague_zhemm != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zhemm);
        dplasma_progress(dague);
        dplasma_zhemm_Destruct( dague_zhemm );
    }
}
