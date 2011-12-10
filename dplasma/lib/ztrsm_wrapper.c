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
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/dplasmatypes.h"

#include "ztrsm_LLN.h"
#include "ztrsm_LLT.h"
#include "ztrsm_LUN.h"
#include "ztrsm_LUT.h"
#include "ztrsm_RLN.h"
#include "ztrsm_RLT.h"
#include "ztrsm_RUN.h"
#include "ztrsm_RUT.h"

dague_object_t *
dplasma_ztrsm_New(const PLASMA_enum side, const PLASMA_enum uplo, 
                  const PLASMA_enum trans, const PLASMA_enum diag,
                  const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B)
{
    dague_object_t *dague_trsm = NULL; 

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
                dague_trsm = (dague_object_t*)dague_ztrsm_LLN_new(
                    side, uplo, trans, diag, alpha,
                    *A, (dague_ddesc_t*)A,
                    *B, (dague_ddesc_t*)B);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_ztrsm_LLT_new(
                    side, uplo, trans, diag, alpha,
                    *A, (dague_ddesc_t*)A,
                    *B, (dague_ddesc_t*)B);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dague_ztrsm_LUN_new(
                    side, uplo, trans, diag, alpha,
                    *A, (dague_ddesc_t*)A,
                    *B, (dague_ddesc_t*)B);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_ztrsm_LUT_new(
                    side, uplo, trans, diag, alpha,
                    *A, (dague_ddesc_t*)A,
                    *B, (dague_ddesc_t*)B);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dague_ztrsm_RLN_new(
                    side, uplo, trans, diag, alpha,
                    *A, (dague_ddesc_t*)A,
                    *B, (dague_ddesc_t*)B);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_ztrsm_RLT_new(
                    side, uplo, trans, diag, alpha,
                    *A, (dague_ddesc_t*)A,
                    *B, (dague_ddesc_t*)B);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trsm = (dague_object_t*)dague_ztrsm_RUN_new(
                    side, uplo, trans, diag, alpha,
                    *A, (dague_ddesc_t*)A,
                    *B, (dague_ddesc_t*)B);
            } else { /* trans =! PlasmaNoTrans */
                dague_trsm = (dague_object_t*)dague_ztrsm_RUT_new(
                    side, uplo, trans, diag, alpha,
                    *A, (dague_ddesc_t*)A,
                    *B, (dague_ddesc_t*)B);
            }
        }
    }

    dplasma_add2arena_tile(((dague_ztrsm_LLN_object_t*)dague_trsm)->arenas[DAGUE_ztrsm_LLN_DEFAULT_ARENA], 
                           A->mb*A->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_trsm;
}

void
dplasma_ztrsm_Destruct( dague_object_t *o )
{
    dague_ztrsm_LLN_object_t *otrsm = (dague_ztrsm_LLN_object_t *)o;
    int side  = ((dague_ztrsm_LLN_object_t *)o)->side;
    int uplo  = ((dague_ztrsm_LLN_object_t *)o)->uplo;
    int trans = ((dague_ztrsm_LLN_object_t *)o)->trans;

    dplasma_datatype_undefine_type( &(otrsm->arenas[DAGUE_ztrsm_LLN_DEFAULT_ARENA]->opaque_dtt) );
    
    if ( side == PlasmaLeft ) {
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_ztrsm_LLN_destroy((dague_ztrsm_LLN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrsm_LLT_destroy((dague_ztrsm_LLT_object_t *)o);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_ztrsm_LUN_destroy((dague_ztrsm_LUN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrsm_LUT_destroy((dague_ztrsm_LUT_object_t *)o);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_ztrsm_RLN_destroy((dague_ztrsm_RLN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrsm_RLT_destroy((dague_ztrsm_RLT_object_t *)o);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_ztrsm_RUN_destroy((dague_ztrsm_RUN_object_t *)o);
            } else { /* trans =! PlasmaNoTrans */
                dague_ztrsm_RUT_destroy((dague_ztrsm_RUT_object_t *)o);
            }
        }
    }
}

void
dplasma_ztrsm( dague_context_t *dague, const PLASMA_enum side, const PLASMA_enum uplo, 
               const PLASMA_enum trans, const PLASMA_enum diag,
               const Dague_Complex64_t alpha, const tiled_matrix_desc_t *A, tiled_matrix_desc_t *B)
{
    dague_object_t *dague_ztrsm = NULL;

    dague_ztrsm = dplasma_ztrsm_New(side, uplo, trans, diag, alpha, A, B);

    if ( dague_ztrsm != NULL ) 
    {
        dague_enqueue( dague, dague_ztrsm );
        dplasma_progress( dague );
        
        dplasma_ztrsm_Destruct( dague_ztrsm );
    }
}
