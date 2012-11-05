/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zsyrk_LN.h"
#include "zsyrk_LT.h"
#include "zsyrk_UN.h"
#include "zsyrk_UT.h"

/***************************************************************************//**
 *
 * @ingroup PLASMA_Complex64_t
 *
 *  dplasm_zsyrk_New - Generates dague object to compute the following operation
 *
 *    \f[ C = \alpha [ op( A ) \times op( A )' ] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X'
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
 *          Specifies whether the matrix A is transposed or transposed:
 *          = PlasmaNoTrans:   A is not transposed;
 *          = PlasmaTrans: A is transposed.
 *
 * @param[in] N
 *          N specifies the order of the matrix C. N must be at least zero.
 *
 * @param[in] K
 *          K specifies the number of columns of the matrix op( A ).
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          A is a LDA-by-ka matrix, where ka is K when trans = PlasmaNoTrans,
 *          and is N otherwise.
 *
 * @param[in] LDA
 *          The leading dimension of the array A. LDA must be at least
 *          max( 1, N ), otherwise LDA must be at least max( 1, K ).
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          C is a LDC-by-N matrix.
 *          On exit, the array uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] LDC
 *          The leading dimension of the array C. LDC >= max( 1, N ).
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *
 *******************************************************************************
 *
 * @sa dplasma_zsyrk
 * @sa dplasma_csyrk_New
 * @sa dplasma_dsyrk_New
 * @sa dplasma_ssyrk_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zsyrk_New( const PLASMA_enum uplo,
                   const PLASMA_enum trans,
                   const dague_complex64_t alpha,
                   const tiled_matrix_desc_t* A,
                   const dague_complex64_t beta,
                   tiled_matrix_desc_t* C)
{
    dague_object_t* object;

    if ( uplo == PlasmaLower ) {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_object_t*)
                dague_zsyrk_LN_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
        else {
            object = (dague_object_t*)
                dague_zsyrk_LT_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
    }
    else {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_object_t*)
                dague_zsyrk_UN_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
        else {
            object = (dague_object_t*)
                dague_zsyrk_UT_new(uplo, trans,
                                   alpha, (dague_ddesc_t*)A,
                                   beta,  (dague_ddesc_t*)C);
        }
    }

    dplasma_add2arena_tile(((dague_zsyrk_LN_object_t*)object)->arenas[DAGUE_zsyrk_LN_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, C->mb);

    return object;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zsyrk_Destruct - Clean the data structures associated to a
 *  zsyrk dague object.
 *
 *******************************************************************************
 *
 * @param[in] o
 *          Object to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zsyrk_New
 * @sa dplasma_zsyrk
 * @sa dplasma_csyrk_Destruct
 * @sa dplasma_dsyrk_Destruct
 * @sa dplasma_ssyrk_Destruct
 *
 ******************************************************************************/
void
dplasma_zsyrk_Destruct( dague_object_t *o )
{
    dague_zsyrk_LN_object_t *zsyrk_object = (dague_zsyrk_LN_object_t*)o;
    dplasma_datatype_undefine_type( &(zsyrk_object->arenas[DAGUE_zsyrk_LN_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(zsyrk_object);
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zsyrk - Synchronous version of dplasma_zsyrk_New
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
 * @sa dplasma_zsyrk_Destruct
 * @sa dplasma_zsyrk_New
 * @sa dplasma_csyrk
 * @sa dplasma_dsyrk
 * @sa dplasma_ssyrk
 *
 ******************************************************************************/
int
dplasma_zsyrk( dague_context_t *dague,
               const PLASMA_enum uplo,
               const PLASMA_enum trans,
               const dague_complex64_t alpha,
               const tiled_matrix_desc_t *A,
               const dague_complex64_t beta,
               tiled_matrix_desc_t *C)
{
    dague_object_t *dague_zsyrk = NULL;

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

    dague_zsyrk = dplasma_zsyrk_New(uplo, trans,
                                    alpha, A,
                                    beta, C);

    if ( dague_zsyrk != NULL )
    {
        dague_enqueue( dague, dague_zsyrk);
        dplasma_progress(dague);
        dplasma_zsyrk_Destruct( dague_zsyrk );
    }
    return 0;
}
