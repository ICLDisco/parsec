/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c
 *
 */
#include "dague_internal.h"
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
 * @ingroup dplasma_complex64_t
 *
 *  dplasm_zherk_New - Generates the object that performs the following operation
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
 *          \retval The dague object describing the operation that can be
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
dague_object_t*
dplasma_zherk_New( PLASMA_enum uplo,
                   PLASMA_enum trans,
                   double alpha,
                   const tiled_matrix_desc_t* A,
                   double beta,
                   tiled_matrix_desc_t* C)
{
    dague_object_t* object;

    if ( uplo == PlasmaLower ) {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_object_t*)
                dague_zherk_LN_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
        else {
            object = (dague_object_t*)
                dague_zherk_LC_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
    }
    else {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_object_t*)
                dague_zherk_UN_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
        else {
            object = (dague_object_t*)
                dague_zherk_UC_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
    }

    dplasma_add2arena_tile(((dague_zherk_LN_object_t*)object)->arenas[DAGUE_zherk_LN_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, C->mb);

    return object;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zherk_Destruct - Free the data structure associated to an object
 *  created with dplasma_zherk_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zherk_New
 * @sa dplasma_zherk
 *
 ******************************************************************************/
void
dplasma_zherk_Destruct( dague_object_t *o )
{
    dague_zherk_LN_object_t *zherk_object = (dague_zherk_LN_object_t*)o;
    dplasma_datatype_undefine_type( &(zherk_object->arenas[DAGUE_zherk_LN_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(zherk_object);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
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
    dague_object_t *dague_zherk = NULL;

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
