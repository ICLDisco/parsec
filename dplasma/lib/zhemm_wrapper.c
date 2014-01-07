/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> c
 *
 */
#include "dague_internal.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zhemm.h"

/**
 ******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zhemm_New - Generates the dague object to compute the following
 *  operation.  WARNING: The computations are not done by this call.
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
 *          Descriptor of the hermitian matrix A. A is a ka-by-ka
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
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zhemm_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zhemm
 * @sa dplasma_zhemm_Destruct
 * @sa dplasma_chemm_New
 *
 ******************************************************************************/
dague_object_t*
dplasma_zhemm_New( PLASMA_enum side,
                   PLASMA_enum uplo,
                   dague_complex64_t alpha,
                   const tiled_matrix_desc_t* A,
                   const tiled_matrix_desc_t* B,
                   dague_complex64_t beta,
                   tiled_matrix_desc_t* C)
{
    dague_zhemm_object_t* object;

    object = dague_zhemm_new(side, uplo, alpha, beta,
                             (dague_ddesc_t*)A,
                             (dague_ddesc_t*)B,
                             (dague_ddesc_t*)C);

    dplasma_add2arena_tile(object->arenas[DAGUE_zhemm_DEFAULT_ARENA],
                           C->mb*C->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, C->mb);

    return (dague_object_t*)object;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zhemm_Destruct - Free the data structure associated to an object
 *  created with dplasma_zhemm_New().
 *
 *******************************************************************************
 *
 * @param[in] o
 *          On entry, the object to destroy
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zhemm_New
 * @sa dplasma_zhemm
 *
 ******************************************************************************/
void
dplasma_zhemm_Destruct( dague_object_t *o )
{
    dague_zhemm_object_t *zhemm_object = (dague_zhemm_object_t*)o;
    dplasma_datatype_undefine_type( &(zhemm_object->arenas[DAGUE_zhemm_DEFAULT_ARENA]->opaque_dtt) );
    DAGUE_INTERNAL_OBJECT_DESTRUCT(zhemm_object);
}

/**
 ******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zhemm - Computes the following operation.
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
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
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
 *          Descriptor of the hermitian matrix A.  A is a ka-by-ka
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
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zhemm_New
 * @sa dplasma_zhemm_Destruct
 * @sa dplasma_chemm
 *
 ******************************************************************************/
int
dplasma_zhemm( dague_context_t *dague,
               PLASMA_enum side,
               PLASMA_enum uplo,
               dague_complex64_t alpha,
               const tiled_matrix_desc_t *A,
               const tiled_matrix_desc_t *B,
               dague_complex64_t beta,
               tiled_matrix_desc_t *C)
{
    dague_object_t *dague_zhemm = NULL;

    /* Check input arguments */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zhemm", "illegal value of side");
        return -1;
    }
    if ((uplo != PlasmaLower) && (uplo != PlasmaUpper)) {
        dplasma_error("dplasma_zhemm", "illegal value of uplo");
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

    dague_zhemm = dplasma_zhemm_New(side, uplo,
                                    alpha, A, B,
                                    beta, C);

    if ( dague_zhemm != NULL )
    {
        dague_enqueue( dague, (dague_object_t*)dague_zhemm);
        dplasma_progress(dague);
        dplasma_zhemm_Destruct( dague_zhemm );
    }
    return 0;
}
