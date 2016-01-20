/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c
 *
 */

#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zherk_LN.h"
#include "zherk_LC.h"
#include "zherk_UN.h"
#include "zherk_UC.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasm_zherk_New - Generates the handle that performs the following operation
 *
 *    \f[ C = \alpha [ op( A ) \times conjg( op( A )' )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = conjg( X' )
 *
 *  where alpha and beta are real scalars, C is an n-by-n hermitian
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
 *          Specifies whether the matrix A is transposed or conjugate transposed:
 *          = PlasmaNoTrans:   A is not transposed;
 *          = PlasmaConjTrans: A is conjugate transposed.
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
 *          \retval The dague handle describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zherk_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zherk
 * @sa dplasma_zherk_Destruct
 * @sa dplasma_cherk_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zherk_New( PLASMA_enum uplo,
                   PLASMA_enum trans,
                   double alpha,
                   const tiled_matrix_desc_t* A,
                   double beta,
                   tiled_matrix_desc_t* C)
{
    dague_handle_t* handle;

    if ( uplo == PlasmaLower ) {
        if ( trans == PlasmaNoTrans ) {
            handle = (dague_handle_t*)
                dague_zherk_LN_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
        else {
            handle = (dague_handle_t*)
                dague_zherk_LC_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
    }
    else {
        if ( trans == PlasmaNoTrans ) {
            handle = (dague_handle_t*)
                dague_zherk_UN_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
        else {
            handle = (dague_handle_t*)
                dague_zherk_UC_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
    }

    dplasma_add2arena_tile(((dague_zherk_LN_handle_t*)handle)->arenas[DAGUE_zherk_LN_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           dague_datatype_double_complex_t, C->mb);

    return handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zherk_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zherk_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zherk_New
 * @sa dplasma_zherk
 *
 ******************************************************************************/
void
dplasma_zherk_Destruct( dague_handle_t *handle )
{
    dague_zherk_LN_handle_t *zherk_handle = (dague_zherk_LN_handle_t*)handle;
    dague_matrix_del2arena( zherk_handle->arenas[DAGUE_zherk_LN_DEFAULT_ARENA] );
    handle->destructor(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasm_zherk - Performs the following operation
 *
 *    \f[ C = \alpha [ op( A ) \times conjg( op( A )' )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = conjg( X' )
 *
 *  where alpha and beta are real scalars, C is an n-by-n hermitian
 *  matrix and A is an n-by-k matrix in the first case and a k-by-n
 *  matrix in the second case.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of C is stored;
 *          = PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is transposed or conjugate transposed:
 *          = PlasmaNoTrans:   A is not transposed;
 *          = PlasmaConjTrans: A is conjugate transposed.
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
 * @sa dplasma_zherk_New
 * @sa dplasma_zherk_Destruct
 * @sa dplasma_cherk
 *
 ******************************************************************************/
int
dplasma_zherk( dague_context_t *dague,
               PLASMA_enum uplo,
               PLASMA_enum trans,
               double alpha,
               const tiled_matrix_desc_t *A,
               double beta,
               tiled_matrix_desc_t *C)
{
    dague_handle_t *dague_zherk = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("PLASMA_zherk", "illegal value of uplo");
        return -1;
    }
    if (trans != PlasmaConjTrans && trans != PlasmaNoTrans ) {
        dplasma_error("dplasma_zherk", "illegal value of trans");
        return -2;
    }
    if ( (C->m != C->n) ) {
        dplasma_error("dplasma_zherk", "illegal size of matrix C which should be square");
        return -6;
    }
    if ( ((trans == PlasmaNoTrans) && (A->m != C->m)) ||
         ((trans != PlasmaNoTrans) && (A->n != C->m)) ) {
        dplasma_error("dplasma_zherk", "illegal size of matrix A");
        return -4;
    }

    dague_zherk = dplasma_zherk_New(uplo, trans,
                                    alpha, A,
                                    beta, C);

    if ( dague_zherk != NULL )
    {
        dague_enqueue( dague, dague_zherk);
        dplasma_progress(dague);
        dplasma_zherk_Destruct( dague_zherk );
    }
    return 0;
}
