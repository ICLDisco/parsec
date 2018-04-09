/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */


#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

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
 *  dplasma_ztrmm_New - Generates parsec taskpool to compute:
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
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
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
parsec_taskpool_t*
dplasma_ztrmm_New( PLASMA_enum side,  PLASMA_enum uplo,
                   PLASMA_enum trans, PLASMA_enum diag,
                   parsec_complex64_t alpha,
                   const parsec_tiled_matrix_dc_t *A,
                   parsec_tiled_matrix_dc_t *B )
{
    parsec_taskpool_t *parsec_trmm = NULL;

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
                parsec_trmm = (parsec_taskpool_t*)parsec_ztrmm_LLN_new(
                    side, uplo, trans, diag, alpha,
                    A, B);
            } else { /* trans =! PlasmaNoTrans */
                parsec_trmm = (parsec_taskpool_t*)parsec_ztrmm_LLT_new(
                    side, uplo, trans, diag, alpha,
                    A, B);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                parsec_trmm = (parsec_taskpool_t*)parsec_ztrmm_LUN_new(
                    side, uplo, trans, diag, alpha,
                    A, B);
            } else { /* trans =! PlasmaNoTrans */
                parsec_trmm = (parsec_taskpool_t*)parsec_ztrmm_LUT_new(
                    side, uplo, trans, diag, alpha,
                    A, B);
            }
        }
    } else { /* side == PlasmaRight */
        if ( uplo == PlasmaLower ) {
            if ( trans == PlasmaNoTrans ) {
                parsec_trmm = (parsec_taskpool_t*)parsec_ztrmm_RLN_new(
                    side, uplo, trans, diag, alpha,
                    A, B);
            } else { /* trans =! PlasmaNoTrans */
                parsec_trmm = (parsec_taskpool_t*)parsec_ztrmm_RLT_new(
                    side, uplo, trans, diag, alpha,
                    A, B);
            }
        } else { /* uplo = PlasmaUpper */
            if ( trans == PlasmaNoTrans ) {
                parsec_trmm = (parsec_taskpool_t*)parsec_ztrmm_RUN_new(
                    side, uplo, trans, diag, alpha,
                    A, B);
            } else { /* trans =! PlasmaNoTrans */
                parsec_trmm = (parsec_taskpool_t*)parsec_ztrmm_RUT_new(
                    side, uplo, trans, diag, alpha,
                    A, B);
            }
        }
    }

    dplasma_add2arena_tile(((parsec_ztrmm_LLN_taskpool_t*)parsec_trmm)->arenas[PARSEC_ztrmm_LLN_DEFAULT_ARENA],
                           A->mb*A->nb*sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_double_complex_t, A->mb);

    return parsec_trmm;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztrmm_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_ztrmm_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztrmm_New
 * @sa dplasma_ztrmm
 *
 ******************************************************************************/
void
dplasma_ztrmm_Destruct( parsec_taskpool_t *tp )
{
    parsec_ztrmm_LLN_taskpool_t *otrmm = (parsec_ztrmm_LLN_taskpool_t *)tp;

    parsec_matrix_del2arena( otrmm->arenas[PARSEC_ztrmm_LLN_DEFAULT_ARENA] );
    parsec_taskpool_free(tp);
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
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
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
dplasma_ztrmm( parsec_context_t *parsec,
               PLASMA_enum side,  PLASMA_enum uplo,
               PLASMA_enum trans, PLASMA_enum diag,
               parsec_complex64_t alpha,
               const parsec_tiled_matrix_dc_t *A,
               parsec_tiled_matrix_dc_t *B)
{
    parsec_taskpool_t *parsec_ztrmm = NULL;

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

    parsec_ztrmm = dplasma_ztrmm_New(side, uplo, trans, diag, alpha, A, B);

    if ( parsec_ztrmm != NULL )
    {
        parsec_enqueue( parsec, (parsec_taskpool_t*)parsec_ztrmm);
        dplasma_wait_until_completion(parsec);
        dplasma_ztrmm_Destruct( parsec_ztrmm );
        return 0;
    }
    else {
        return -101;
    }
}
