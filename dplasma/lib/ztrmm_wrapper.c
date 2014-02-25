/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */


#include <core_blas.h>
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

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztrmm_New - Generates dague object to compute:
 *
 *  B = alpha*op( A )*B or B = alpha*B*op( A ).
 *
 *  WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Specifies whether A appears on the left or on the right of X:
 *          = PlasmaLeft:  A*X = B
 *          = PlasmaRight: X*A = B
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower triangular:
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] trans
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
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          Descriptor of the triangular matrix A of size N-by-N.
 *          The triangular matrix A. If uplo = PlasmaUpper, the leading N-by-N upper triangular
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
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_ztrmm_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrmm
 * @sa dplasma_ztrmm_Destruct
 * @sa dplasma_ctrmm_New
 * @sa dplasma_dtrmm_New
 * @sa dplasma_strmm_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_ztrmm_New( PLASMA_enum side,  PLASMA_enum uplo,
                   PLASMA_enum trans, PLASMA_enum diag,
                   dague_complex64_t alpha,
                   const tiled_matrix_desc_t *A,
                   tiled_matrix_desc_t *B )
{
    dague_handle_t *dague_trmm = NULL;

    /* Check input arguments */
    if (side != PlasmaLeft && side != PlasmaRight) {
        dplasma_error("dplasma_ztrmm_New", "illegal value of side");
        return NULL /*-1*/;
    }
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_ztrmm_New", "illegal value of uplo");
        return NULL /*-2*/;
    }
    if (trans != PlasmaConjTrans && trans != PlasmaNoTrans && trans != PlasmaTrans ) {
        dplasma_error("dplasma_ztrmm_New", "illegal value of trans");
        return NULL /*-3*/;
    }
    if (diag != PlasmaUnit && diag != PlasmaNonUnit) {
        dplasma_error("dplasma_ztrmm_New", "illegal value of diag");
        return NULL /*-4*/;
    }

    if ( side == PlasmaLeft ) {
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_handle_t*)dague_ztrmm_LLN_new(
                    side, uplo, trans, diag, alpha,
                    (dague_ddesc_t*)A, (dague_ddesc_t*)B);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_handle_t*)dague_ztrmm_LLT_new(
                    side, uplo, trans, diag, alpha,
                    (dague_ddesc_t*)A, (dague_ddesc_t*)B);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_handle_t*)dague_ztrmm_LUN_new(
                    side, uplo, trans, diag, alpha,
                    (dague_ddesc_t*)A, (dague_ddesc_t*)B);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_handle_t*)dague_ztrmm_LUT_new(
                    side, uplo, trans, diag, alpha,
                    (dague_ddesc_t*)A, (dague_ddesc_t*)B);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_handle_t*)dague_ztrmm_RLN_new(
                    side, uplo, trans, diag, alpha,
                    (dague_ddesc_t*)A, (dague_ddesc_t*)B);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_handle_t*)dague_ztrmm_RLT_new(
                    side, uplo, trans, diag, alpha,
                    (dague_ddesc_t*)A, (dague_ddesc_t*)B);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                dague_trmm = (dague_handle_t*)dague_ztrmm_RUN_new(
                    side, uplo, trans, diag, alpha,
                    (dague_ddesc_t*)A, (dague_ddesc_t*)B);
            } else { /* trans =! PlasmaNoTrans */
                dague_trmm = (dague_handle_t*)dague_ztrmm_RUT_new(
                    side, uplo, trans, diag, alpha,
                    (dague_ddesc_t*)A, (dague_ddesc_t*)B);
            }
        }
    }

    dplasma_add2arena_tile(((dague_ztrmm_LLN_handle_t*)dague_trmm)->arenas[DAGUE_ztrmm_LLN_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return dague_trmm;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztrmm_Destruct - Free the data structure associated to an object
 *  created with dplasma_ztrmm_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrmm_New
 * @sa dplasma_ztrmm
 *
 ******************************************************************************/
void
dplasma_ztrmm_Destruct( dague_handle_t *o )
{
    dague_ztrmm_LLN_handle_t *otrmm = (dague_ztrmm_LLN_handle_t *)o;

    dplasma_datatype_undefine_type( &(otrmm->arenas[DAGUE_ztrmm_LLN_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztrmm - Computes:
 *
 *  B = alpha*op( A )*B or B = alpha*B*op( A ).
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] side
 *          Specifies whether A appears on the left or on the right of X:
 *          = PlasmaLeft:  A*X = B
 *          = PlasmaRight: X*A = B
 *
 * @param[in] uplo
 *          Specifies whether the matrix A is upper triangular or lower triangular:
 *          = PlasmaUpper: Upper triangle of A is stored;
 *          = PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] trans
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
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          Descriptor of the triangular matrix A of size N-by-N.
 *          The triangular matrix A. If uplo = PlasmaUpper, the leading N-by-N upper triangular
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
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrmm
 * @sa dplasma_ztrmm_Destruct
 * @sa dplasma_ctrmm_New
 * @sa dplasma_dtrmm_New
 * @sa dplasma_strmm_New
 *
 ******************************************************************************/
int
dplasma_ztrmm( dague_context_t *dague,
               PLASMA_enum side,  PLASMA_enum uplo,
               PLASMA_enum trans, PLASMA_enum diag,
               dague_complex64_t alpha,
               const tiled_matrix_desc_t *A,
               tiled_matrix_desc_t *B)
{
    dague_handle_t *dague_ztrmm = NULL;

    /* Check input arguments */
    if (side != PlasmaLeft && side != PlasmaRight) {
        dplasma_error("dplasma_ztrmm", "illegal value of side");
        return -1;
    }
    if (uplo != PlasmaUpper && uplo != PlasmaLower) {
        dplasma_error("dplasma_ztrmm", "illegal value of uplo");
        return -2;
    }
    if (trans != PlasmaConjTrans && trans != PlasmaNoTrans && trans != PlasmaTrans ) {
        dplasma_error("dplasma_ztrmm", "illegal value of trans");
        return -3;
    }
    if (diag != PlasmaUnit && diag != PlasmaNonUnit) {
        dplasma_error("dplasma_ztrmm", "illegal value of diag");
        return -4;
    }

    if ( (A->m != A->n) ||
         (( side == PlasmaLeft )  && (A->n != B->m)) ||
         (( side == PlasmaRight ) && (A->n != B->n)) ) {
        dplasma_error("dplasma_ztrmm_New", "illegal matrix A");
        return -6;
    }

    dague_ztrmm = dplasma_ztrmm_New(side, uplo, trans, diag, alpha, A, B);

    if ( dague_ztrmm != NULL )
    {
        dague_enqueue( dague, (dague_handle_t*)dague_ztrmm);
        dplasma_progress(dague);
        dplasma_ztrmm_Destruct( dague_ztrmm );
        return 0;
    }
    else {
        return -101;
    }
}
