/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> c
 *
 */
#include "dague_internal.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zher2k_LN.h"
#include "zher2k_LC.h"
#include "zher2k_UN.h"
#include "zher2k_UC.h"

/***************************************************************************//**
 *
 * @ingroup PLASMA_Complex64_t
 *
 *  dplasm_zher2k_New - Performs one of the hermitian rank 2k operations
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
 * @param[in] uplo
 *          = PlasmaUpper: Upper triangle of C is stored;
 *          = PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is transposed or conjugate transposed:
 *          = PlasmaNoTrans:   \f[ C = \alpha [ op( A ) \times conjg( op( B )' )] + conjg( \alpha ) [ op( B ) \times conjg( op( A )' )] + \beta C \f]
 *          = PlasmaConjTrans: \f[ C = \alpha [ conjg( op( A )' ) \times op( B ) ] + conjg( \alpha ) [ conjg( op( B )' ) \times op( A ) ] + \beta C \f]
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
 * @sa dplasma_zher2k
 * @sa dplasma_cher2k_New
 * @sa dplasma_dher2k_New
 * @sa dplasma_sher2k_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zher2k_New( const PLASMA_enum uplo,
                    const PLASMA_enum trans,
                    const double alpha,
                    const tiled_matrix_desc_t* A,
                    const tiled_matrix_desc_t* B,
                    const double beta,
                    tiled_matrix_desc_t* C)
{
    dague_object_t* object;

    if ( uplo == PlasmaLower ) {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_object_t*)
                dague_zher2k_LN_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
        else {
            object = (dague_object_t*)
                dague_zher2k_LT_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
    }
    else {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_object_t*)
                dague_zher2k_UN_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
        else {
            object = (dague_object_t*)
                dague_zher2k_UN_new(uplo, trans,
                                    alpha, (dague_ddesc_t*)A,
                                           (dague_ddesc_t*)B,
                                    beta,  (dague_ddesc_t*)C);
        }
    }

    dplasma_add2arena_tile(((dague_zher2k_LN_object_t*)object)->arenas[DAGUE_zher2k_LN_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, C->mb);

    return object;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zher2k_Destruct - Clean the data structures associated to a
 *  zher2k dague object.
 *
 *******************************************************************************
 *
 * @param[in] o
 *          Object to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zher2k_New
 * @sa dplasma_zher2k
 * @sa dplasma_cher2k_Destruct
 * @sa dplasma_dher2k_Destruct
 * @sa dplasma_sher2k_Destruct
 *
 ******************************************************************************/
void
dplasma_zher2k_Destruct( dague_object_t *o )
{
    dague_zher2k_LN_object_t *zher2k_object = (dague_zher2k_LN_object_t*)o;
    dplasma_datatype_undefine_type( &(zher2k_object->arenas[DAGUE_zher2k_LN_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(zher2k_object);
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zher2k - Synchronous version of dplasma_zher2k_New
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
 * @sa dplasma_zher2k_Destruct
 * @sa dplasma_zher2k_New
 * @sa dplasma_cher2k
 * @sa dplasma_dher2k
 * @sa dplasma_sher2k
 *
 ******************************************************************************/
int
dplasma_zher2k( dague_context_t *dague,
                const PLASMA_enum uplo,
                const PLASMA_enum trans,
                const double alpha,
                const tiled_matrix_desc_t *A,
                const tiled_matrix_desc_t *B,
                const double beta,
                tiled_matrix_desc_t *C)
{
    dague_object_t *dague_zher2k = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("PLASMA_zher2k", "illegal value of uplo");
        return -1;
    }
    if (trans != PlasmaConjTrans && trans != PlasmaNoTrans ) {
        dplasma_error("dplasma_zher2k", "illegal value of trans");
        return -2;
    }

    dague_zher2k = dplasma_zher2k_New(uplo, trans,
                                      alpha, A, B,
                                      beta, C);

    if ( dague_zher2k != NULL )
    {
        dague_enqueue( dague, dague_zher2k);
        dplasma_progress(dague);
        dplasma_zher2k_Destruct( dague_zher2k );
    }
    return 0;
}
