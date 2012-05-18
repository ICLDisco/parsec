/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
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
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#include "ztrmm_LLN.h"
#include "ztrmm_LLT.h"
#include "ztrmm_LUN.h"
#include "ztrmm_LUT.h"
#include "ztrmm_RLN.h"
#include "ztrmm_RLT.h"
#include "ztrmm_RUN.h"
#include "ztrmm_RUT.h"

dague_object_t *
dplasma_ztrmm_New( const PLASMA_enum side, const PLASMA_enum uplo, 
                   const PLASMA_enum trans, const PLASMA_enum diag,
                   const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B)
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
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dague_ztrmm_LLT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dague_ztrmm_LUN_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dague_ztrmm_LUT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dague_ztrmm_RLN_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dague_ztrmm_RLT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_object_t*)dague_ztrmm_RUN_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A,
                    side, uplo, trans, diag, alpha,
                    A->m, A->n, A->mb, A->nb, A->mt, A->nt,
                    B->m, B->n, B->mb, B->nb, B->mt, B->nt);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_object_t*)dague_ztrmm_RUT_new(
                    (dague_ddesc_t*)B, (dague_ddesc_t*)A,
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

    return dague_trmm;
}

void
dplasma_ztrmm_Destruct( dague_object_t *o )
{
    dague_ztrmm_LLN_object_t *otrmm = (dague_ztrmm_LLN_object_t *)o;

    dplasma_datatype_undefine_type( &(otrmm->arenas[DAGUE_ztrmm_LLN_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

void
dplasma_ztrmm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, const PLASMA_enum trans, const PLASMA_enum diag,
               const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B)
{
    dague_object_t *dague_ztrmm = NULL;

    dague_ztrmm = dplasma_ztrmm_New(side, uplo, trans, diag, alpha, A, B);

    if ( dague_ztrmm != NULL )
    {
      dague_enqueue( dague, (dague_object_t*)dague_ztrmm);
      dplasma_progress(dague);
      
      dplasma_ztrmm_Destruct( dague_ztrmm );
    }
}
