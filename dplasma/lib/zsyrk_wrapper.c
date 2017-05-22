/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */

#include <core_blas.h>
#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zsyrk_LN.h"
#include "zsyrk_LT.h"
#include "zsyrk_UN.h"
#include "zsyrk_UT.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasm_zsyrk_New - Generates the taskpool that performs the following operation
 *
 *    \f[ C = \alpha [ op( A ) \times op( A )' ] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X'
 *
 *  where alpha and beta are real scalars, C is an n-by-n symmetric
 *  matrix and A is an n-by-k matrix in the first case and a k-by-n
 *  matrix in the second case.
 *
 *  WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of C is stored;
 *          = PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is transposed or transposed:
 *          = PlasmaNoTrans: A is not transposed;
 *          = PlasmaTrans:   A is transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          A is a LDA-by-ka matrix, where ka is K when trans = PlasmaNoTrans,
 *          and is N otherwise.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          C is a LDC-by-N matrix.
 *          On exit, the array uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zsyrk_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zsyrk
 * @sa dplasma_csyrk_New
 * @sa dplasma_dsyrk_New
 * @sa dplasma_ssyrk_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zsyrk_New( PLASMA_enum uplo,
                   PLASMA_enum trans,
                   parsec_complex64_t alpha,
                   const tiled_matrix_desc_t* A,
                   parsec_complex64_t beta,
                   tiled_matrix_desc_t* C)
{
    parsec_taskpool_t* tp;

    if ( uplo == PlasmaLower ) {
        if ( trans == PlasmaNoTrans ) {
            tp = (parsec_taskpool_t*)
                parsec_zsyrk_LN_new(uplo, trans,
                                   alpha, A,
                                   beta,  C);
        }
        else {
            tp = (parsec_taskpool_t*)
                parsec_zsyrk_LT_new(uplo, trans,
                                   alpha, A,
                                   beta,  C);
        }
    }
    else {
        if ( trans == PlasmaNoTrans ) {
            tp = (parsec_taskpool_t*)
                parsec_zsyrk_UN_new(uplo, trans,
                                   alpha, A,
                                   beta,  C);
        }
        else {
            tp = (parsec_taskpool_t*)
                parsec_zsyrk_UT_new(uplo, trans,
                                   alpha, A,
                                   beta,  C);
        }
    }

    dplasma_add2arena_tile(((parsec_zsyrk_LN_taskpool_t*)tp)->arenas[PARSEC_zsyrk_LN_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_double_complex_t, C->mb);

    return tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zsyrk_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zsyrk_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zsyrk_New
 * @sa dplasma_zsyrk
 *
 ******************************************************************************/
void
dplasma_zsyrk_Destruct( parsec_taskpool_t *tp )
{
    parsec_zsyrk_LN_taskpool_t *zsyrk_tp = (parsec_zsyrk_LN_taskpool_t*)tp;
    parsec_matrix_del2arena( zsyrk_tp->arenas[PARSEC_zsyrk_LN_DEFAULT_ARENA] );
    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasm_zsyrk - Performs the following operation
 *
 *    \f[ C = \alpha [ op( A ) \times op( A )' ] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = conjg( X' )
 *
 *  where alpha and beta are real scalars, C is an n-by-n symmetric
 *  matrix and A is an n-by-k matrix in the first case and a k-by-n
 *  matrix in the second case.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of C is stored;
 *          = PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is transposed or transposed:
 *          = PlasmaNoTrans: A is not transposed;
 *          = PlasmaTrans:   A is transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          A is a LDA-by-ka matrix, where ka is K when trans = PlasmaNoTrans,
 *          and is N otherwise.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          C is a LDC-by-N matrix.
 *          On exit, the array uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zsyrk_New
 * @sa dplasma_zsyrk_Destruct
 * @sa dplasma_csyrk
 * @sa dplasma_dsyrk
 * @sa dplasma_ssyrk
 *
 ******************************************************************************/
int
dplasma_zsyrk( parsec_context_t *parsec,
               PLASMA_enum uplo,
               PLASMA_enum trans,
               parsec_complex64_t alpha,
               const tiled_matrix_desc_t *A,
               parsec_complex64_t beta,
               tiled_matrix_desc_t *C)
{
    parsec_taskpool_t *parsec_zsyrk = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("PLASMA_zsyrk", "illegal value of uplo");
        return -1;
    }
    if (trans != PlasmaTrans && trans != PlasmaNoTrans ) {
        dplasma_error("dplasma_zsyrk", "illegal value of trans");
        return -2;
    }
    if ( (C->m != C->n) ) {
        dplasma_error("dplasma_zsyrk", "illegal size of matrix C which should be square");
        return -6;
    }
    if ( ((trans == PlasmaNoTrans) && (A->m != C->m)) ||
         ((trans != PlasmaNoTrans) && (A->n != C->m)) ) {
        dplasma_error("dplasma_zsyrk", "illegal size of matrix A");
        return -4;
    }

    parsec_zsyrk = dplasma_zsyrk_New(uplo, trans,
                                    alpha, A,
                                    beta, C);

    if ( parsec_zsyrk != NULL )
    {
        parsec_enqueue( parsec, parsec_zsyrk);
        dplasma_wait_until_completion(parsec);
        dplasma_zsyrk_Destruct( parsec_zsyrk );
    }
    return 0;
}
