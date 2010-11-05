/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include <plasma.h>
#include <dague.h>
#include <scheduling.h>
#include "dplasma.h"
#include "dplasmatypes.h"

#include "generated/zgemm_NN.h"
#include "generated/zgemm_NT.h"
#include "generated/zgemm_TN.h"
#include "generated/zgemm_TT.h"

dague_object_t*
dplasma_zgemm_New( const int transA, const int transB,
                   const Dague_Complex64_t alpha, const tiled_matrix_desc_t* A, const tiled_matrix_desc_t* B,
                   const Dague_Complex64_t beta,  tiled_matrix_desc_t* C)
{
    dague_object_t* zgemm_object;
    dague_arena_t* arena;

    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            dague_zgemm_NN_object_t* object;
            object = dague_zgemm_NN_new((dague_ddesc_t*)C, (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                                        transA, transB, alpha, beta,
                                        A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                                        B->m, B->n, B->mb, B->nb, B->mt, B->nt,
                                        C->m, C->n, C->mb, C->nb, C->mt, C->nt);
            arena = object->arenas[DAGUE_zgemm_NN_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        } else {
            dague_zgemm_NT_object_t* object;
            object = dague_zgemm_NT_new((dague_ddesc_t*)C, (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                                        transA, transB, alpha, beta,
                                        A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                                        B->m, B->n, B->mb, B->nb, B->mt, B->nt,
                                        C->m, C->n, C->mb, C->nb, C->mt, C->nt);
            arena = object->arenas[DAGUE_zgemm_NT_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        }
    } else {
        if( PlasmaNoTrans == transB ) {
            dague_zgemm_TN_object_t* object;
            object = dague_zgemm_TN_new((dague_ddesc_t*)C, (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                                        transA, transB, alpha, beta,
                                        A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                                        B->m, B->n, B->mb, B->nb, B->mt, B->nt,
                                        C->m, C->n, C->mb, C->nb, C->mt, C->nt);
            arena = object->arenas[DAGUE_zgemm_TN_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        } else {
            dague_zgemm_TT_object_t* object;
            object = dague_zgemm_TT_new((dague_ddesc_t*)C, (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                                        transA, transB, alpha, beta,
                                        A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                                        B->m, B->n, B->mb, B->nb, B->mt, B->nt,
                                        C->m, C->n, C->mb, C->nb, C->mt, C->nt);
            arena = object->arenas[DAGUE_zgemm_TT_DEFAULT_ARENA];
            zgemm_object = (dague_object_t*)object;
        }
    }

    dplasma_add2arena_tile(arena, 
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return zgemm_object;
}

void
dplasma_zgemm_Destruct( dague_object_t *o )
{
    int transA = ((dague_zgemm_NN_object_t *)o)->transA;
    int transB = ((dague_zgemm_NN_object_t *)o)->transB;

    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            dague_zgemm_NN_destroy((dague_zgemm_NN_object_t *)o);
        } else {
            dague_zgemm_NT_destroy((dague_zgemm_NT_object_t *)o);
        }
    } else {
        if( PlasmaNoTrans == transB ) {
            dague_zgemm_TN_destroy((dague_zgemm_TN_object_t *)o);
        } else {
            dague_zgemm_TT_destroy((dague_zgemm_TT_object_t *)o);
        }
    }
}

void
dplasma_zgemm( dague_context_t *dague, const int transA, const int transB,
               const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, const tiled_matrix_desc_t *B,
               const Dague_Complex64_t beta,  tiled_matrix_desc_t *C)
{
    dague_object_t *dague_zgemm = NULL;

    dague_zgemm = dplasma_zgemm_New(transA, transB, 
                                    alpha, A, B,
                                    beta, C);

    dague_enqueue( dague, (dague_object_t*)dague_zgemm);
    dague_progress(dague);
    dplasma_zgemm_Destruct( dague_zgemm );
}
