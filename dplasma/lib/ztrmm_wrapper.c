/*
 * Copyright (c) 2010-2011 The University of Tennessee and The University
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
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "generated/ztrmm_LLN.h"
#include "generated/ztrmm_LLT.h"
#include "generated/ztrmm_LUN.h"
#include "generated/ztrmm_LUT.h"
#include "generated/ztrmm_RLN.h"
#include "generated/ztrmm_RLT.h"
#include "generated/ztrmm_RUN.h"
#include "generated/ztrmm_RUT.h"

dague_object_t *
dplasma_ztrmm_New( const PLASMA_enum side, const PLASMA_enum uplo, 
                   const PLASMA_enum trans, const PLASMA_enum diag,
                   const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B, tiled_matrix_desc_t *work)
{
    dague_object_t *dague_trmm = NULL;

    /* Check input arguments */
    if (side != PlasmaLeft && side != PlasmaRight) {
        dplasma_error("dplasma_ztrsm_New", "illegal value of side");
        return NULL /*-1*/;
    }
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_ztrsm_New", "illegal value of uplo");
        return NULL /*-2*/;
    }
    if (trans != PlasmaConjTrans && trans != PlasmaNoTrans && trans != PlasmaTrans ) {
        dplasma_error("dplasma_ztrsm_New", "illegal value of trans");
        return NULL /*-3*/;
    }
    if (diag != PlasmaUnit && diag != PlasmaNonUnit) {
        dplasma_error("dplasma_ztrsm_New", "illegal value of diag");
        return NULL /*-4*/;
    }

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
#if 0
    dplasma_add2arena_tile(((dague_ztrmm_LLN_object_t*)dague_trmm)->arenas[DAGUE_ztrmm_LLN_CONTROL_ARENA], 
                           sizeof(int),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_INTEGER, 1);
#else
    ((dague_ztrmm_LLN_object_t*)dague_trmm)->arenas[DAGUE_ztrmm_LLN_CONTROL_ARENA] = NULL;
#endif

    return dague_trmm;
}

void
dplasma_ztrmm_Destruct( dague_object_t *o )
{
    int side  = ((dague_ztrmm_LLN_object_t *)o)->side;
    int uplo  = ((dague_ztrmm_LLN_object_t *)o)->uplo;
    int trans = ((dague_ztrmm_LLN_object_t *)o)->trans;

    if ( side == PlasmaLeft ) {
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_ztrmm_LLN_destroy((dague_ztrmm_LLN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrmm_LLT_destroy((dague_ztrmm_LLT_object_t *)o);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_ztrmm_LUN_destroy((dague_ztrmm_LUN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrmm_LUT_destroy((dague_ztrmm_LUT_object_t *)o);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_ztrmm_RLN_destroy((dague_ztrmm_RLN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrmm_RLT_destroy((dague_ztrmm_RLT_object_t *)o);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_ztrmm_RUN_destroy((dague_ztrmm_RUN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrmm_RUT_destroy((dague_ztrmm_RUT_object_t *)o);
            }
        }
    }
}

void
dplasma_ztrmm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
               const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B)
{
    dague_object_t *dague_ztrmm = NULL;

    two_dim_block_cyclic_t work;
    /* Create workspace for control */
    two_dim_block_cyclic_init(&work, matrix_Integer, B->super.nodes, B->super.cores, B->super.myrank,
                              1, 1, B->mt, B->nt, 0, 0, B->mt, B->nt, 1, 1, ((two_dim_block_cyclic_t*)B)->grid.rows);
    work.mat = dague_data_allocate((size_t)work.super.nb_local_tiles * (size_t)work.super.bsiz * (size_t)work.super.mtype);
    
    dague_ztrmm = dplasma_ztrmm_New(side, uplo, trans, diag, alpha,
				    A, B, (tiled_matrix_desc_t *)&work);

    if ( dague_ztrmm != NULL )
    {
      dague_enqueue( dague, (dague_object_t*)dague_ztrmm);
      dague_progress(dague);
      
      dplasma_ztrmm_Destruct( dague_ztrmm );
    }

    dague_data_free(work.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&work);
}
