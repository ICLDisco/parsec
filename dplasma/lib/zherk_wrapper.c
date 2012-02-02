/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague.h"
#include <plasma.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zherk_LN.h"
#include "zherk_LT.h"
#include "zherk_UN.h"
#include "zherk_UT.h"

/***************************************************************************//**
 *
 * @ingroup PLASMA_Complex64_t
 *
 *  dplasm_zherk_New - Generates dague object to compute the following operation
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
 * @sa dplasma_zherk
 * @sa dplasma_cherk_New
 * @sa dplasma_dherk_New
 * @sa dplasma_sherk_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zherk_New( const PLASMA_enum uplo, 
                   const PLASMA_enum trans,
                   const double alpha, 
                   const tiled_matrix_desc_t* A, 
                   const double beta, 
                   tiled_matrix_desc_t* C)
{
    dague_object_t* object;

    if ( uplo == PlasmaLower ) {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_object_t*)
                dague_zherk_LN_new(uplo, trans,
                                   alpha, *A, (dague_ddesc_t*)A, 
                                   beta,  *C, (dague_ddesc_t*)C);
        }
        else {
            object = (dague_object_t*)
                dague_zherk_LT_new(uplo, trans,
                                   alpha, *A, (dague_ddesc_t*)A, 
                                   beta,  *C, (dague_ddesc_t*)C);
        }
    }
    else {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_object_t*)
                dague_zherk_UN_new(uplo, trans,
                                   alpha, *A, (dague_ddesc_t*)A, 
                                   beta,  *C, (dague_ddesc_t*)C);
        }
        else {
            object = (dague_object_t*)
                dague_zherk_UN_new(uplo, trans,
                                   alpha, *A, (dague_ddesc_t*)A, 
                                   beta,  *C, (dague_ddesc_t*)C);
        }
    }

    dplasma_add2arena_tile(((dague_zherk_LN_object_t*)object)->arenas[DAGUE_zherk_LN_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(Dague_Complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, C->mb);

    return object;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zherk_Destruct - Clean the data structures associated to a
 *  zherk dague object.
 *
 *******************************************************************************
 *
 * @param[in] o
 *          Object to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zherk_New
 * @sa dplasma_zherk
 * @sa dplasma_cherk_Destruct
 * @sa dplasma_dherk_Destruct
 * @sa dplasma_sherk_Destruct
 *
 ******************************************************************************/
void
dplasma_zherk_Destruct( dague_object_t *o )
{
    dague_zherk_LN_object_t *zherk_object = (dague_zherk_LN_object_t*)o;
    dplasma_datatype_undefine_type( &(zherk_object->arenas[DAGUE_zherk_LN_DEFAULT_ARENA]->opaque_dtt) );
    dague_zherk_destroy(zherk_object);
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zherk - Synchronous version of dplasma_zherk_New
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
 * @sa dplasma_zherk_Destruct
 * @sa dplasma_zherk_New
 * @sa dplasma_cherk
 * @sa dplasma_dherk
 * @sa dplasma_sherk
 *
 ******************************************************************************/
int
dplasma_zherk( dague_context_t *dague, 
               const PLASMA_enum uplo, 
               const PLASMA_enum trans,
               const double alpha, 
               const tiled_matrix_desc_t *A, 
               const double beta, 
               tiled_matrix_desc_t *C)
{
    dague_object_t *dague_zherk = NULL;

    /* Check input arguments */
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("PLASMA_zherk", "illegal value of uplo");
        return -1;
    }
    if (trans != PlasmaConjTrans && trans != PlasmaNoTrans && trans != PlasmaTrans ) {
        dplasma_error("dplasma_zherk", "illegal value of trans");
        return -2;
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
