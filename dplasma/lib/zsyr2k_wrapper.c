/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> z c d s
 *
 */

#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zsyr2k_LN.h"
#include "zsyr2k_LT.h"
#include "zsyr2k_UN.h"
#include "zsyr2k_UT.h"

/**
 ******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zsyr2k_New - Generates the dague object to performs one of the
 *  syrmitian rank 2k operations
 *
 *    \f[ C = \alpha [ op( A ) \times conjg( op( B )' )] + conjg( \alpha ) [ op( B ) \times conjg( op( A )' )] + \beta C \f],
 *    or
 *    \f[ C = \alpha [ conjg( op( A )' ) \times op( B ) ] + conjg( \alpha ) [ conjg( op( B )' ) \times op( A ) ] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = conjg( X' )
 *
 *  where alpha and beta are real scalars, C is an n-by-n symmetric
 *  matrix and A and B are an n-by-k matrices the first case and k-by-n
 *  matrices in the second case.
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
 *          Specifies whether the matrix A is not transposed or conjugate transposed:
 *          = PlasmaNoTrans: \f[ C = \alpha [ op( A )  \times op( B )' ] + conjg( \alpha ) [ op( B )  \times op( A )' ] + \beta C \f]
 *          = PlasmaTrans:   \f[ C = \alpha [ op( A )' \times op( B )  ] + conjg( \alpha ) [ op( B )' \times op( A )  ] + \beta C \f]
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B
 *
 * @param[in] beta
 *          beta specifies the scalar beta.
 *
 * @param[in,out] C
 *          Descriptor of the symmetric matrix C.
 *          On exit, the uplo part of the matrix described by C is overwritten
 *          by the result of the operation.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zsyr2k_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zsyr2k
 * @sa dplasma_zsyr2k_Destruct
 * @sa dplasma_csyr2k_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zsyr2k_New( PLASMA_enum uplo,
                    PLASMA_enum trans,
                    dague_complex64_t alpha,
                    const tiled_matrix_desc_t* A,
                    const tiled_matrix_desc_t* B,
                    dague_complex64_t beta,
                    tiled_matrix_desc_t* C)
{
    dague_handle_t* object;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("dplasma_zsyr2k_New", "illegal value of uplo");
        return NULL;
    }
    if (trans != PlasmaConjTrans && trans != PlasmaTrans && trans != PlasmaNoTrans ) {
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
            object = (dague_handle_t*)
                dague_zsyr2k_LN_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
        else {
            object = (dague_handle_t*)
                dague_zsyr2k_LT_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
    }
    else {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_handle_t*)
                dague_zsyr2k_UN_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
        else {
            object = (dague_handle_t*)
                dague_zsyr2k_UT_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
    }

    dplasma_add2arena_tile(((dague_zsyr2k_LN_handle_t*)object)->arenas[DAGUE_zsyr2k_LN_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, C->mb);

    return object;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zsyr2k_Destruct - Free the data structure associated to an object
 *  created with dplasma_zsyr2k_New().
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
 *
 ******************************************************************************/
void
dplasma_zsyr2k_Destruct( dague_handle_t *o )
{
    dague_zsyr2k_LN_handle_t *zsyr2k_object = (dague_zsyr2k_LN_handle_t*)o;
    dplasma_datatype_undefine_type( &(zsyr2k_object->arenas[DAGUE_zsyr2k_LN_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_HANDLE_DESTRUCT(zsyr2k_object);
}

/**
 ******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zsyr2k - Performs one of the symmetric rank 2k operations
 *
 *    \f[ C = \alpha [ op( A ) \times conjg( op( B )' )] + conjg( \alpha ) [ op( B ) \times conjg( op( A )' )] + \beta C \f],
 *    or
 *    \f[ C = \alpha [ conjg( op( A )' ) \times op( B ) ] + conjg( \alpha ) [ conjg( op( B )' ) \times op( A ) ] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = conjg( X' )
 *
 *  where alpha and beta are real scalars, C is an n-by-n symmetric
 *  matrix and A and B are an n-by-k matrices the first case and k-by-n
 *  matrices in the second case.
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
 *          Specifies whether the matrix A is not transposed or conjugate transposed:
 *          = PlasmaNoTrans:   \f[ C = \alpha [ op( A ) \times conjg( op( B )' )] + conjg( \alpha ) [ op( B ) \times conjg( op( A )' )] + \beta C \f]
 *          = PlasmaConjTrans: \f[ C = \alpha [ conjg( op( A )' ) \times op( B ) ] + conjg( \alpha ) [ conjg( op( B )' ) \times op( A ) ] + \beta C \f]
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B
 *
 * @param[in] beta
 *          beta specifies the scalar beta.
 *
 * @param[in,out] C
 *          Descriptor of the symmetric matrix C.
 *          On exit, the uplo part of the matrix described by C is overwritten
 *          by the result of the operation.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zsyr2k_New
 * @sa dplasma_zsyr2k_Destruct
 * @sa dplasma_csyr2k
 *
 ******************************************************************************/
int
dplasma_zsyr2k( dague_context_t *dague,
                PLASMA_enum uplo,
                PLASMA_enum trans,
                dague_complex64_t alpha,
                const tiled_matrix_desc_t *A,
                const tiled_matrix_desc_t *B,
                dague_complex64_t beta,
                tiled_matrix_desc_t *C)
{
    dague_handle_t *dague_zsyr2k = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("dplasma_zsyr2k", "illegal value of uplo");
        return -1;
    }
    if (trans != PlasmaConjTrans && trans != PlasmaTrans && trans != PlasmaNoTrans ) {
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
    if ( (( trans == PlasmaNoTrans ) && ( A->m != C->m )) ||
         (( trans != PlasmaNoTrans ) && ( A->n != C->m )) ) {
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
