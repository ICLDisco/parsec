/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */
#include "dague_internal.h"
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zsyr2k_LN.h"
#include "zsyr2k_LT.h"
#include "zsyr2k_UN.h"
#include "zsyr2k_UT.h"

/***************************************************************************//**
 *
 * @ingroup PLASMA_Complex64_t
 *
 *  dplasma_zsyr2k_New - Performs one of the symmetric rank 2k operations
 *
 *    \f[ C = \alpha [ op( A ) \times conjg( op( B )' )] + \alpha [ op( B ) \times conjg( op( A )' )] + \beta C \f],
 *    or
 *    \f[ C = \alpha [ conjg( op( A )' ) \times op( B ) ] + \alpha [ conjg( op( B )' ) \times op( A ) ] + \beta C \f],
 *
 *  wsyre op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X'
 *
 *  where alpha and beta are real scalars, C is an n-by-n symmetric
 *  matrix and A and B are an n-by-k matrices the first case and k-by-n
 *  matrices in the second case.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of C is stored;
 *          = PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is not transposed or transposed:
 *          = PlasmaNoTrans:   \f[ C = \alpha [ op( A ) \times conjg( op( B )' )] + \alpha [ op( B ) \times conjg( op( A )' )] + \beta C \f]
 *          = PlasmaTrans: \f[ C = \alpha [ conjg( op( A )' ) \times op( B ) ] + \alpha [ conjg( op( B )' ) \times op( A ) ] + \beta C \f]
 *
 * @param[in] N
 *          N specifies the order of the matrix C. N must be at least zero.
 *
 * @param[in] K
 *          K specifies the number of columns of the A and B matrices with trans = PlasmaNoTrans.
 *          K specifies the number of rows of the A and B matrices with trans = PlasmaTrans.
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
 * @param[in] B
 *          B is a LDB-by-kb matrix, where kb is K when trans = PlasmaNoTrans,
 *          and is N otherwise.
 *
 * @param[in] LDB
 *          The leading dimension of the array B. LDB must be at least
 *          max( 1, N ), otherwise LDB must be at least max( 1, K ).
 *
 * @param[in] beta
 *          beta specifies the scalar beta.
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
 * @sa dplasma_zsyr2k
 * @sa dplasma_csyr2k_New
 * @sa dplasma_dsyr2k_New
 * @sa dplasma_ssyr2k_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zsyr2k_New( const PLASMA_enum uplo,
                    const PLASMA_enum trans,
                    const dague_complex64_t alpha,
                    const tiled_matrix_desc_t* A,
                    const tiled_matrix_desc_t* B,
                    const dague_complex64_t beta,
                    tiled_matrix_desc_t* C)
{
    dague_object_t* object;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("dplasma_zsyr2k_New", "illegal value of uplo");
        return NULL;
    }
    if (trans != PlasmaTrans && trans != PlasmaNoTrans ) {
        dplasma_error("dplasma_zsyr2k_New", "illegal value of trans");
        return NULL;
    }

    if ( C->m != C->n ) {
        dplasma_error("dplasma_zsyr2k_New", "illegal descriptor C (C->m != C->n)");
        return NULL;
    }
    if ( A->m != B->m || A->n != B->n ) {
        dplasma_error("dplasma_zsyr2k_New", "illegal descriptor A or B, they must have the same dimensions");
        return NULL;
    }
    if ( (( trans == PlasmaNoTrans ) && ( A->m != C->m ))
         || (( trans != PlasmaNoTrans ) && ( A->n != C->m )) ) {
        dplasma_error("dplasma_zsyr2k_New", "illegal sizes for the matrices");
        return NULL;
    }

    if ( uplo == PlasmaLower ) {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_object_t*)
                dague_zsyr2k_LN_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
        else {
            object = (dague_object_t*)
                dague_zsyr2k_LT_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
    }
    else {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_object_t*)
                dague_zsyr2k_UN_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
        else {
            object = (dague_object_t*)
                dague_zsyr2k_UT_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
    }

    dplasma_add2arena_tile(((dague_zsyr2k_LN_object_t*)object)->arenas[DAGUE_zsyr2k_LN_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, C->mb);

    return object;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zsyr2k_Destruct - Clean the data structures associated to a
 *  zsyr2k dague object.
 *
 *******************************************************************************
 *
 * @param[in] o
 *          Object to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zsyr2k_New
 * @sa dplasma_zsyr2k
 * @sa dplasma_csyr2k_Destruct
 * @sa dplasma_dsyr2k_Destruct
 * @sa dplasma_ssyr2k_Destruct
 *
 ******************************************************************************/
void
dplasma_zsyr2k_Destruct( dague_object_t *o )
{
    dague_zsyr2k_LN_object_t *zsyr2k_object = (dague_zsyr2k_LN_object_t*)o;
    dplasma_datatype_undefine_type( &(zsyr2k_object->arenas[DAGUE_zsyr2k_LN_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(zsyr2k_object);
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zsyr2k - Synchronous version of dplasma_zsyr2k_New
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
 * @sa dplasma_zsyr2k_Destruct
 * @sa dplasma_zsyr2k_New
 * @sa dplasma_csyr2k
 * @sa dplasma_dsyr2k
 * @sa dplasma_ssyr2k
 *
 ******************************************************************************/
int
dplasma_zsyr2k( dague_context_t *dague,
                const PLASMA_enum uplo,
                const PLASMA_enum trans,
                const dague_complex64_t alpha,
                const tiled_matrix_desc_t *A,
                const tiled_matrix_desc_t *B,
                const dague_complex64_t beta,
                tiled_matrix_desc_t *C)
{
    dague_object_t *dague_zsyr2k = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("dplasma_zsyr2k", "illegal value of uplo");
        return -1;
    }
    if (trans != PlasmaTrans && trans != PlasmaNoTrans ) {
        dplasma_error("dplasma_zsyr2k", "illegal value of trans");
        return -2;
    }

    if ( A->m != B->m || A->n != B->n ) {
        dplasma_error("dplasma_zsyr2k", "illegal descriptor A or B, they must have the same dimensions");
        return -4;
    }
    if ( C->m != C->n ) {
        dplasma_error("dplasma_zsyr2k", "illegal descriptor C (C->m != C->n)");
        return -6;
    }
    if ( (( trans == PlasmaNoTrans ) && ( A->m != C->m ))
         || (( trans != PlasmaNoTrans ) && ( A->n != C->m )) ) {
        dplasma_error("dplasma_zsyr2k", "illegal sizes for the matrices");
        return -6;
    }

    dague_zsyr2k = dplasma_zsyr2k_New(uplo, trans,
                                      alpha, A, B,
                                      beta, C);

    if ( dague_zsyr2k != NULL )
    {
        dague_enqueue( dague, dague_zsyr2k);
        dplasma_progress(dague);
        dplasma_zsyr2k_Destruct( dague_zsyr2k );
    }
    return 0;
}
