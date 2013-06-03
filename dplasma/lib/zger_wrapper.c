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
#include "dplasma/lib/dplasmatypes.h"

#include "zger.h"

static inline dague_object_t*
dplasma_zger_internal_New( int trans, dague_complex64_t alpha,
                           const tiled_matrix_desc_t *X,
                           const tiled_matrix_desc_t *Y,
                           tiled_matrix_desc_t *A)
{
    dague_zger_object_t* zger_object;

    /* Check input arguments */
    if ((trans != PlasmaTrans) && (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zger", "illegal value of trans");
        return NULL /*-1*/;
    }
    zger_object = dague_zger_new(trans, alpha,
                                 (dague_ddesc_t*)X,
                                 (dague_ddesc_t*)Y,
                                 (dague_ddesc_t*)A);

    dplasma_add2arena_tile( zger_object->arenas[DAGUE_zger_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb);

    dplasma_add2arena_tile( zger_object->arenas[DAGUE_zger_VECTOR_ARENA],
                            A->mb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb);

    return (dague_object_t*)zger_object;
}

static inline void
dplasma_zger_internal_Destruct( dague_object_t *o )
{
    dplasma_datatype_undefine_type( &(((dague_zger_object_t *)o)->arenas[DAGUE_zger_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(((dague_zger_object_t *)o)->arenas[DAGUE_zger_VECTOR_ARENA]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

static inline int
dplasma_zger_internal( dague_context_t *dague,
                       const int trans,
                       const dague_complex64_t alpha,
                       const tiled_matrix_desc_t *X,
                       const tiled_matrix_desc_t *Y,
                             tiled_matrix_desc_t *A)
{
    dague_object_t *dague_zger = NULL;

    /* Check input arguments */
    if ((trans != PlasmaTrans) && (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zger", "illegal value of trans");
        return -1;
    }

    dague_zger = dplasma_zger_internal_New(trans, alpha, X, Y, A);

    if ( dague_zger != NULL )
    {
        dague_enqueue( dague, dague_zger);
        dplasma_progress(dague);
        dplasma_zger_internal_Destruct( dague_zger );
        return 0;
    }
    else {
        return -101;
    }
}

dague_object_t*
dplasma_zgeru_New( const dague_complex64_t alpha,
                   const tiled_matrix_desc_t *X,
                   const tiled_matrix_desc_t *Y,
                         tiled_matrix_desc_t *A)
{
    return dplasma_zger_internal_New( PlasmaTrans, alpha, X, Y, A );
}

void
dplasma_zgeru_Destruct( dague_object_t *o )
{
    dplasma_zger_internal_Destruct( o );
}

int
dplasma_zgeru( dague_context_t *dague,
               const dague_complex64_t alpha,
               const tiled_matrix_desc_t *X,
               const tiled_matrix_desc_t *Y,
                     tiled_matrix_desc_t *A)
{
    return dplasma_zger_internal( dague, PlasmaTrans, alpha, X, Y, A );
}

#if defined(PRECISION_z) || defined(PRECISION_c)

dague_object_t*
dplasma_zgerc_New( const dague_complex64_t alpha,
                   const tiled_matrix_desc_t *X,
                   const tiled_matrix_desc_t *Y,
                         tiled_matrix_desc_t *A)
{
    return dplasma_zger_internal_New( PlasmaConjTrans, alpha, X, Y, A );
}

void
dplasma_zgerc_Destruct( dague_object_t *o )
{
    dplasma_zger_internal_Destruct( o );
}

int
dplasma_zgerc( dague_context_t *dague,
               const dague_complex64_t alpha,
               const tiled_matrix_desc_t *X,
               const tiled_matrix_desc_t *Y,
                     tiled_matrix_desc_t *A)
{
    return dplasma_zger_internal( dague, PlasmaConjTrans, alpha, X, Y, A );
}

#endif /* defined(PRECISION_z) || defined(PRECISION_c) */
