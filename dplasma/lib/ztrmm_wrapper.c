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
#include "data_dist/matrix/two_dim_rectangle_cyclic/two_dim_rectangle_cyclic.h"

#include "generated/ztrmm_LLN.h"
#include "generated/ztrmm_LLT.h"
#include "generated/ztrmm_LUN.h"
#include "generated/ztrmm_LUT.h"
#include "generated/ztrmm_RLN.h"
#include "generated/ztrmm_RLT.h"
#include "generated/ztrmm_RUN.h"
#include "generated/ztrmm_RUT.h"

dague_object_t *
dplasma_ztrmm_New( const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
                   const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *work)
{
    dague_object_t *dague_trmm = NULL;

    if ( side == PlasmaLeft ) {
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dague_ztrmm_LLN_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, (dague_ddesc_t*)work,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dague_ztrmm_LLT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, (dague_ddesc_t*)work,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dague_ztrmm_LUN_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, (dague_ddesc_t*)work,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dague_ztrmm_LUT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, (dague_ddesc_t*)work,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dague_ztrmm_RLN_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, (dague_ddesc_t*)work,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dague_ztrmm_RLT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, (dague_ddesc_t*)work,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dague_ztrmm_RUN_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, (dague_ddesc_t*)work,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dague_ztrmm_RUT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A, (dague_ddesc_t*)work,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            }
        }
    }

    dplasma_add2arena_tile(((dague_ztrmm_LLN_object_t*)dague_trmm)->arenas[DAGUE_ztrmm_LLN_DEFAULT_ARENA], 
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    dplasma_add2arena_tile(((dague_ztrmm_LLN_object_t*)dague_trmm)->arenas[DAGUE_ztrmm_LLN_CONTROL_ARENA], 
                           sizeof(int),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_INTEGER, 1);

    return dague_trmm;
}

void
dplasma_ztrmm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
               const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B)
{
    dague_object_t *dague_ztrmm = NULL;

    two_dim_block_cyclic_t work;
    /* Create workspace for control */
    two_dim_block_cyclic_init(&work, matrix_Integer, B->super.nodes, B->super.cores, B->super.myrank,
                              1, 1, B->mt, B->nt, 0, 0, B->mt, B->nt, 1, 1, ((two_dim_block_cyclic_t*)B)->GRIDrows);
    work.mat = dague_data_allocate((size_t)work.super.nb_local_tiles * (size_t)work.super.bsiz * (size_t)work.super.mtype);
    
    dague_ztrmm = dplasma_ztrmm_New(side, uplo, trans, diag, alpha,
				    A, B, (tiled_matrix_desc_t *)&work);

    dague_enqueue( dague, (dague_object_t*)dague_ztrmm);

    fprintf(stderr, "Nb tasks to do : %d\n", dague->taskstodo);

    dague_progress(dague);

    dague_data_free(work.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&work);
}
