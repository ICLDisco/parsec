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

#include "zgemm_NN.h"
#include "zgemm_NT.h"
#include "zgemm_TN.h"
#include "zgemm_TT.h"

dague_object_t*
dplasma_zgemm_New( const int transA, const int transB,
                   const dague_complex64_t alpha, const tiled_matrix_desc_t* A, const tiled_matrix_desc_t* B,
                   const dague_complex64_t beta,  tiled_matrix_desc_t* C)
{
    dague_object_t* zgemm_object;
    dague_arena_t* arena;

    /* Check input arguments */
    if ((transA != PlasmaNoTrans) && (transA != PlasmaTrans) && (transA != PlasmaConjTrans)) {
        dplasma_error("PLASMA_zgemm", "illegal value of transA");
        return NULL /*-1*/;
    }
    if ((transB != PlasmaNoTrans) && (transB != PlasmaTrans) && (transB != PlasmaConjTrans)) {
        dplasma_error("PLASMA_zgemm", "illegal value of transB");
        return NULL /*-2*/;
    }

    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            dague_zgemm_NN_object_t* object;
            object = dague_zgemm_NN_new(transA, transB, alpha, beta,
                                        *A, (dague_ddesc_t*)A,
                                        *B, (dague_ddesc_t*)B,
                                        *C, (dague_ddesc_t*)C);
            arena = object->arenas[DAGUE_zgemm_NN_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        } else {
            dague_zgemm_NT_object_t* object;
            object = dague_zgemm_NT_new(transA, transB, alpha, beta,
                                        *A, (dague_ddesc_t*)A,
                                        *B, (dague_ddesc_t*)B,
                                        *C, (dague_ddesc_t*)C);
            arena = object->arenas[DAGUE_zgemm_NT_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        }
    } else {
        if( PlasmaNoTrans == transB ) {
            dague_zgemm_TN_object_t* object;
            object = dague_zgemm_TN_new(transA, transB, alpha, beta,
                                        *A, (dague_ddesc_t*)A,
                                        *B, (dague_ddesc_t*)B,
                                        *C, (dague_ddesc_t*)C);
            arena = object->arenas[DAGUE_zgemm_TN_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        } else {
            dague_zgemm_TT_object_t* object;
            object = dague_zgemm_TT_new(transA, transB, alpha, beta,
                                        *A, (dague_ddesc_t*)A,
                                        *B, (dague_ddesc_t*)B,
                                        *C, (dague_ddesc_t*)C);
            arena = object->arenas[DAGUE_zgemm_TT_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        }
    }

    dplasma_add2arena_tile(arena,
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return zgemm_object;
}

void
dplasma_zgemm_Destruct( dague_object_t *o )
{
    dplasma_datatype_undefine_type( &(((dague_zgemm_NN_object_t *)o)->arenas[DAGUE_zgemm_NN_DEFAULT_ARENA]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

void
dplasma_zgemm( dague_context_t *dague, const int transA, const int transB,
               const dague_complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
               const dague_complex64_t beta,  tiled_matrix_desc_t *C)
{
    dague_object_t *dague_zgemm = NULL;

    dague_zgemm = dplasma_zgemm_New(transA, transB,
                                    alpha, A, B,
                                    beta, C);

    if ( dague_zgemm != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zgemm);
        dplasma_progress(dague);
        dplasma_zgemm_Destruct( dague_zgemm );
    }
}
