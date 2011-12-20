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

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_ztrsm_New - Generates dague object to compute triangular solve 
 *     op( A ) * X = B or X * op( A ) = B
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Specifies whether A appears on the left or on the right of X:
 *          = PlasmaLeft:  op( A ) * X = B
 *          = PlasmaRight: X * op( A ) = B
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower triangular:
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or conjugate transposed:
 *          = PlasmaNoTrans:   A is transposed;
 *          = PlasmaTrans:     A is not transposed;
 *          = PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] diag
 *          Specifies whether or not A is unit triangular:
 *          = PlasmaNonUnit: A is non unit;
 *          = PlasmaUnit:    A us unit.
 *
 * @param[in] A
 *          Descriptor of the triangular matrix A of size N-by-N.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular
 *          part of the array A contains the upper triangular matrix, and the strictly lower
 *          triangular part of A is not referenced. If uplo = PlasmaLower, the leading N-by-N
 *          lower triangular part of the array A contains the lower triangular matrix, and the
 *          strictly upper triangular part of A is not referenced. If diag = PlasmaUnit, the
 *          diagonal elements of A are also not referenced and are assumed to be 1.
 *
 * @param[in,out] B
 *          Descriptor of the N-by-NRHS right hand side B
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if return value = 0, the N-by-NRHS solution matrix X.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 0 successful exit
 *          \retval <0 if -i, the i-th argument had an illegal value
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrsm
 * @sa dplasma_ztrsm_Destruct
 * @sa dplasma_ctrsm
 * @sa dplasma_dtrsm
 * @sa dplasma_strsm
 *
 ******************************************************************************/
dague_object_t*
dplasma_ztrsm_New(const PLASMA_enum side,  const PLASMA_enum uplo, 
                  const PLASMA_enum trans, const PLASMA_enum diag,
                  const Dague_Complex64_t alpha, 
                  const tiled_matrix_desc_t *A, 
                  tiled_matrix_desc_t *B )
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

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_ztrsm_Destruct - Clean the data structures associated to a
 *  ztrsm dague object.
 *
 *******************************************************************************
 *
 * @param[in] o
 *          Object to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrsm_New
 * @sa dplasma_ztrsm
 * @sa dplasma_ctrsm_Destruct
 * @sa dplasma_dtrsm_Destruct
 * @sa dplasma_strsm_Destruct
 *
 ******************************************************************************/
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

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_ztrsm - Synchronous version of dplasma_ztrsm_New
 *
 *******************************************************************************
 *
 * @param[in] dague
 *          Dague context to which submit the DAG object.
 *
 *******************************************************************************
 *
 * @return
 *          \retval 0 if success
 *          \retval < 0 if one of the parameter had an illegal value.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrsm_Destruct
 * @sa dplasma_ztrsm_New
 * @sa dplasma_ctrsm
 * @sa dplasma_dtrsm
 * @sa dplasma_strsm
 *
 ******************************************************************************/
int 
void
dplasma_ztrsm( dague_context_t *dague, 
               const PLASMA_enum side, const PLASMA_enum uplo, 
               const PLASMA_enum trans, const PLASMA_enum diag,
               const Dague_Complex64_t alpha, 
               const tiled_matrix_desc_t *A, 
               tiled_matrix_desc_t *B)
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
