/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> c
 *
 */
#include "dague_internal.h"
#include <core_blas.h>
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zsymm.h"

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zsymm_New - Generates dague object to compute the following operation
 *
 *     \f[ C = \alpha \times A \times B + \beta \times C \f]
 *
 *  or
 *
 *     \f[ C = \alpha \times B \times A + \beta \times C \f]
 *
 *  where alpha and beta are scalars, A is an hermitian matrix and  B and
 *  C are m by n matrices.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          Specifies whether the hermitian matrix A appears on the
 *          left or right in the operation as follows:
 *          = PlasmaLeft:      \f[ C = \alpha \times A \times B + \beta \times C \f]
 *          = PlasmaRight:     \f[ C = \alpha \times B \times A + \beta \times C \f]
 *
 * @param[in] uplo
 *          Specifies whether the upper or lower triangular part of
 *          the hermitian matrix A is to be referenced as follows:
 *          = PlasmaLower:     Only the lower triangular part of the
 *                             hermitian matrix A is to be referenced.
 *          = PlasmaUpper:     Only the upper triangular part of the
 *                             hermitian matrix A is to be referenced.
 *
 * @param[in] alpha
 *          Specifies the scalar alpha.
 *
 * @param[in] A
 *          Descriptor of the triangular matrix A.  A is a ka-by-ka
 *          matrix, where ka is C->M when side = PlasmaLeft, and is
 *          C->N otherwise. Only the uplo triangular part is
 *          referenced.
 *
 * @param[in] B
 *          Descriptor of the M-by-N matrix B
 *
 * @param[in] beta
 *          Specifies the scalar beta.
 *
 * @param[in,out] C
 *          Descriptor of the M-by-N matrix C which is overwritten by
 *          the result of the operation.
 *
 *******************************************************************************
 *
 * @return the dague object describing the operation.
 *
 *******************************************************************************
 *
 * @sa dplasma_zsymm
 * @sa dplasma_zsymm_Destruct
 * @sa dplasma_csymm
 * @sa dplasma_dsymm
 * @sa dplasma_ssymm
 *
 ******************************************************************************/
dague_object_t*
dplasma_zsymm_New( const PLASMA_enum side,
                   const PLASMA_enum uplo,
                   const dague_complex64_t alpha,
                   const tiled_matrix_desc_t* A,
                   const tiled_matrix_desc_t* B,
                   const dague_complex64_t beta,
                   tiled_matrix_desc_t* C)
{
    dague_zsymm_object_t* object;

    object = dague_zsymm_new(side, uplo, alpha, beta,
                             *A, (dague_ddesc_t*)A,
                             *B, (dague_ddesc_t*)B,
                             *C, (dague_ddesc_t*)C);

    dplasma_add2arena_tile(object->arenas[DAGUE_zsymm_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, C->mb);

    return (dague_object_t*)object;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zsymm_Destruct - Clean the data structures associated to a
 *  zsymm dague object.
 *
 *******************************************************************************
 *
 * @param[in] o
 *          Object to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zsymm_New
 * @sa dplasma_zsymm
 * @sa dplasma_csymm_Destruct
 * @sa dplasma_dsymm_Destruct
 * @sa dplasma_ssymm_Destruct
 *
 ******************************************************************************/
void
dplasma_zsymm_Destruct( dague_object_t *o )
{
    dague_zsymm_object_t *zsymm_object = (dague_zsymm_object_t*)o;
    dplasma_datatype_undefine_type( &(zsymm_object->arenas[DAGUE_zsymm_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(zsymm_object);
}

/***************************************************************************//**
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zsymm - Synchronous version of dplasma_zsymm_New
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
 * @sa dplasma_zsymm_Destruct
 * @sa dplasma_zsymm_New
 * @sa dplasma_csymm
 * @sa dplasma_dsymm
 * @sa dplasma_ssymm
 *
 ******************************************************************************/
int
dplasma_zsymm( dague_context_t *dague,
               const PLASMA_enum side,
               const PLASMA_enum uplo,
               const dague_complex64_t alpha,
               const tiled_matrix_desc_t *A,
               const tiled_matrix_desc_t *B,
               const dague_complex64_t beta,
               tiled_matrix_desc_t *C)
{
    dague_object_t *dague_zsymm = NULL;

    /* Check input arguments */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zsymm", "illegal value of side");
        return -1;
    }
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("dplasma_zsymm", "illegal value of uplo");
        return -2;
    }
    if ( (A->m != A->n) ) {
        dplasma_error("dplasma_zhemm", "illegal size of matrix A which should be square");
        return -4;
    }
    if ( (B->m != C->m) || (B->n != C->n) ) {
        dplasma_error("dplasma_zhemm", "illegal sizes of matrices B and C");
        return -5;
    }
    if ( ((side == PlasmaLeft) && (A->n != C->m)) ||
         ((side == PlasmaRight) && (A->n != C->n)) ) {
        dplasma_error("dplasma_zhemm", "illegal size of matrix A");
        return -6;
    }

    dague_zsymm = dplasma_zsymm_New(side, uplo,
                                    alpha, A, B,
                                    beta, C);

    if ( dague_zsymm != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zsymm);
        dplasma_progress(dague);
        dplasma_zsymm_Destruct( dague_zsymm );
    }
    return 0;
}
