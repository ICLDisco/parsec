/*
 * Copyright (c) 2010-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> z c
 *
 */

#include "dplasma.h"
#include "dplasmatypes.h"

#include "zher2k_LN.h"
#include "zher2k_LC.h"
#include "zher2k_UN.h"
#include "zher2k_UC.h"

/**
 ******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zher2k_New - Generates the parsec taskpool to performs one of the
 *  hermitian rank 2k operations
 *
 *    \f[ C = \alpha [ op( A ) \times conjg( op( B )' )] + conjg( \alpha ) [ op( B ) \times conjg( op( A )' )] + \beta C \f],
 *    or
 *    \f[ C = \alpha [ conjg( op( A )' ) \times op( B ) ] + conjg( \alpha ) [ conjg( op( B )' ) \times op( A ) ] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = conjg( X' )
 *
 *  where alpha and beta are real scalars, C is an n-by-n hermitian
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
 *          Descriptor of the hermitian matrix C.
 *          On exit, the uplo part of the matrix described by C is overwritten
 *          by the result of the operation.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_zher2k_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zher2k
 * @sa dplasma_zher2k_Destruct
 * @sa dplasma_cher2k_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zher2k_New( PLASMA_enum uplo,
                    PLASMA_enum trans,
                    parsec_complex64_t alpha,
                    const parsec_tiled_matrix_dc_t* A,
                    const parsec_tiled_matrix_dc_t* B,
                    double beta,
                    parsec_tiled_matrix_dc_t* C)
{
    parsec_taskpool_t* tp;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("dplasma_zher2k_New", "illegal value of uplo");
        return NULL;
    }
    if (trans != PlasmaConjTrans && trans != PlasmaNoTrans ) {
        dplasma_error("dplasma_zher2k_New", "illegal value of trans");
        return NULL;
    }

    if ( C->m != C->n ) {
        dplasma_error("dplasma_zher2k_New", "illegal descriptor C (C->m != C->n)");
        return NULL;
    }
    if ( A->m != B->m || A->n != B->n ) {
        dplasma_error("dplasma_zher2k_New", "illegal descriptor A or B, they must have the same dimensions");
        return NULL;
    }
    if ( (( trans == PlasmaNoTrans ) && ( A->m != C->m ))
         || (( trans != PlasmaNoTrans ) && ( A->n != C->m )) ) {
        dplasma_error("dplasma_zher2k_New", "illegal sizes for the matrices");
        return NULL;
    }

    if ( uplo == PlasmaLower ) {
        if ( trans == PlasmaNoTrans ) {
            tp = (parsec_taskpool_t*)
                parsec_zher2k_LN_new(uplo, trans,
                                    alpha, A,
                                           B,
                                    beta,  C);
        }
        else {
            tp = (parsec_taskpool_t*)
                parsec_zher2k_LC_new(uplo, trans,
                                    alpha, A,
                                           B,
                                    beta,  C);
        }
    }
    else {
        if ( trans == PlasmaNoTrans ) {
            tp = (parsec_taskpool_t*)
                parsec_zher2k_UN_new(uplo, trans,
                                    alpha, A,
                                           B,
                                    beta,  C);
        }
        else {
            tp = (parsec_taskpool_t*)
                parsec_zher2k_UC_new(uplo, trans,
                                    alpha, A,
                                           B,
                                    beta,  C);
        }
    }

    dplasma_add2arena_tile(((parsec_zher2k_LN_taskpool_t*)tp)->arenas[PARSEC_zher2k_LN_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(parsec_complex64_t),
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           parsec_datatype_double_complex_t, C->mb);

    return tp;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zher2k_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zher2k_New().
 *
 *******************************************************************************
 *
 * @param[in] tp
 *          taskpool to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zher2k_New
 * @sa dplasma_zher2k
 *
 ******************************************************************************/
void
dplasma_zher2k_Destruct( parsec_taskpool_t *tp )
{
    parsec_zher2k_LN_taskpool_t *zher2k_tp = (parsec_zher2k_LN_taskpool_t*)tp;
    parsec_matrix_del2arena( zher2k_tp->arenas[PARSEC_zher2k_LN_DEFAULT_ARENA] );
    parsec_taskpool_free(tp);
}

/**
 ******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zher2k - Performs one of the hermitian rank 2k operations
 *
 *    \f[ C = \alpha [ op( A ) \times conjg( op( B )' )] + conjg( \alpha ) [ op( B ) \times conjg( op( A )' )] + \beta C \f],
 *    or
 *    \f[ C = \alpha [ conjg( op( A )' ) \times op( B ) ] + conjg( \alpha ) [ conjg( op( B )' ) \times op( A ) ] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = conjg( X' )
 *
 *  where alpha and beta are real scalars, C is an n-by-n hermitian
 *  matrix and A and B are an n-by-k matrices the first case and k-by-n
 *  matrices in the second case.
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
 *          Descriptor of the hermitian matrix C.
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
 * @sa dplasma_zher2k_New
 * @sa dplasma_zher2k_Destruct
 * @sa dplasma_cher2k
 *
 ******************************************************************************/
int
dplasma_zher2k( parsec_context_t *parsec,
                PLASMA_enum uplo,
                PLASMA_enum trans,
                parsec_complex64_t alpha,
                const parsec_tiled_matrix_dc_t *A,
                const parsec_tiled_matrix_dc_t *B,
                double beta,
                parsec_tiled_matrix_dc_t *C)
{
    parsec_taskpool_t *parsec_zher2k = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("dplasma_zher2k", "illegal value of uplo");
        return -1;
    }
    if (trans != PlasmaConjTrans && trans != PlasmaNoTrans ) {
        dplasma_error("dplasma_zher2k", "illegal value of trans");
        return -2;
    }

    if ( A->m != B->m || A->n != B->n ) {
        dplasma_error("dplasma_zher2k", "illegal descriptor A or B, they must have the same dimensions");
        return -4;
    }
    if ( C->m != C->n ) {
        dplasma_error("dplasma_zher2k", "illegal descriptor C (C->m != C->n)");
        return -6;
    }
    if ( (( trans == PlasmaNoTrans ) && ( A->m != C->m )) ||
         (( trans != PlasmaNoTrans ) && ( A->n != C->m )) ) {
        dplasma_error("dplasma_zher2k", "illegal sizes for the matrices");
        return -6;
    }

    parsec_zher2k = dplasma_zher2k_New(uplo, trans,
                                      alpha, A, B,
                                      beta, C);

    if ( parsec_zher2k != NULL )
    {
        parsec_context_add_taskpool( parsec, parsec_zher2k);
        dplasma_wait_until_completion(parsec);
        dplasma_zher2k_Destruct( parsec_zher2k );
    }
    return 0;
}
