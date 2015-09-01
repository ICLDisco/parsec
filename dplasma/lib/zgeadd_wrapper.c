/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> c d s
 *
 */

#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/cores/dplasma_zcores.h"

#include "map2.h"

typedef struct ztradd_args_s {
    PLASMA_enum       trans;
    dague_complex64_t alpha;
    dague_complex64_t beta;
} ztradd_args_t;

static int
dplasma_ztradd_operator( dague_execution_unit_t *eu,
                         const tiled_matrix_desc_t *descA,
                         const tiled_matrix_desc_t *descB,
                         const void *_A, void *_B,
                         PLASMA_enum uplo, int m, int n,
                         void *args )
{
    const dague_complex64_t *A     = (dague_complex64_t*)_A;
    dague_complex64_t       *B     = (dague_complex64_t*)_B;
    ztradd_args_t           *_args = (ztradd_args_t*)args;
    PLASMA_enum              trans = _args->trans;
    dague_complex64_t        alpha = _args->alpha;
    dague_complex64_t        beta  = _args->beta;

    int tempmm, tempnn, ldam, ldbm;
    (void)eu;

    tempmm = ((m)==((descB->mt)-1)) ? ((descB->m)-(m*(descB->mb))) : (descB->mb);
    tempnn = ((n)==((descB->nt)-1)) ? ((descB->n)-(n*(descB->nb))) : (descB->nb);
    if (trans == PlasmaNoTrans) {
        ldam = BLKLDD( *descA, m );
    }
    else {
        ldam = BLKLDD( *descA, n );
    }
    ldbm = BLKLDD( *descB, m );

    return dplasma_core_ztradd( uplo, trans, tempmm, tempnn,
                                alpha, A, ldam, beta, B, ldbm );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_ztradd_New - Generates an object that computes the operation B =
 * alpha * op(A) + beta * B, where op(A) is one of op(A) = A or op(A) = A' or
 * op(A) = conj(A')
 * A and B are upper or lower trapezoidal matricesn or general matrices.
 * This function combines both pztradd and pzgeadd functionnalities from PBLAS
 * library.
 *
 * See dplasma_map2_New() for further information.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies the shape of A and B matrices:
 *          = PlasmaUpperLower: A and B are general matrices.
 *          = PlasmaUpper: op(A) and B are upper trapezoidal matrices.
 *          = PlasmaLower: op(A) and B are lower trapezoidal matrices.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is non-transposed, transposed, or
 *          conjugate transposed
 *          = PlasmaNoTrans:   op(A) = A
 *          = PlasmaTrans:     op(A) = A'
 *          = PlasmaConjTrans: op(A) = conj(A')
 *
 * @param[in] alpha
 *          The scalar alpha
 *
 * @param[in] A
 *          The descriptor of the distributed matrix A.
 *
 * @param[in] beta
 *          The scalar beta
 *
 * @param[in,out] B
 *          The descriptor of the distributed matrix B of size M-by-N.
 *          On exit, the matrix B data are overwritten by the result of
 *          alpha * op(A) + beta * B
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_ztradd_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_ztradd
 * @sa dplasma_ztradd_Destruct
 * @sa dplasma_ctradd_New
 * @sa dplasma_dtradd_New
 * @sa dplasma_stradd_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_ztradd_New( PLASMA_enum uplo, PLASMA_enum trans,
                    dague_complex64_t alpha,
                    const tiled_matrix_desc_t *A,
                    dague_complex64_t beta,
                    tiled_matrix_desc_t *B)
{
    ztradd_args_t *args = (ztradd_args_t*)malloc(sizeof(ztradd_args_t));
    args->trans = trans;
    args->alpha = alpha;
    args->beta  = beta;

    return dplasma_map2_New( uplo, trans, A, B,
                             dplasma_ztradd_operator, (void *)args );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeadd_New - Generates an object that computes the operation B =
 * alpha * op(A) + beta * B, where op(A) is one of op(A) = A or op(A) = A' or
 * op(A) = conj(A')
 * A and B are general matrices.
 *
 * See dplasma_ztradd_New() for further information.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] trans
 *          Specifies whether the matrix A is non-transposed, transposed, or
 *          conjugate transposed
 *          = PlasmaNoTrans:   op(A) = A
 *          = PlasmaTrans:     op(A) = A'
 *          = PlasmaConjTrans: op(A) = conj(A')
 *
 * @param[in] alpha
 *          The scalar alpha
 *
 * @param[in] A
 *          The descriptor of the distributed matrix A.
 *
 * @param[in] beta
 *          The scalar beta
 *
 * @param[in,out] B
 *          The descriptor of the distributed matrix B of size M-by-N.
 *          On exit, the matrix B data are overwritten by the result of
 *          alpha * op(A) + beta * B
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgeadd_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeadd
 * @sa dplasma_zgeadd_Destruct
 * @sa dplasma_cgeadd_New
 * @sa dplasma_dgeadd_New
 * @sa dplasma_sgeadd_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zgeadd_New( PLASMA_enum trans,
                    dague_complex64_t alpha,
                    const tiled_matrix_desc_t *A,
                    dague_complex64_t beta,
                    tiled_matrix_desc_t *B)
{
    return dplasma_ztradd_New( PlasmaUpperLower, trans, alpha, A, beta, B );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_ztradd_Destruct - Free the data structure associated to an object
 *  created with dplasma_ztradd_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztradd_New
 * @sa dplasma_ztradd
 *
 ******************************************************************************/
void
dplasma_ztradd_Destruct( dague_handle_t *o )
{
    dplasma_map2_Destruct( o );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeadd_Destruct - Free the data structure associated to an object
 *  created with dplasma_zgeadd_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeadd_New
 * @sa dplasma_zgeadd
 *
 ******************************************************************************/
void
dplasma_zgeadd_Destruct( dague_handle_t *o )
{
    dplasma_ztradd_Destruct( o );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_ztradd - Generates an object that computes the operation B = alpha *
 * op(A) + beta * B, and op(A) is one of op(A) = A, or op(A) = A', or op(A) =
 * conj(A')
 *
 * See dplasma_map2() for further information.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies the shape of A and B matrices:
 *          = PlasmaUpperLower: A and B are general matrices.
 *          = PlasmaUpper: op(A) and B are upper trapezoidal matrices.
 *          = PlasmaLower: op(A) and B are lower trapezoidal matrices.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is non-transposed, transposed, or
 *          conjugate transposed
 *          = PlasmaNoTrans:   op(A) = A
 *          = PlasmaTrans:     op(A) = A'
 *          = PlasmaConjTrans: op(A) = conj(A')
 *
 * @param[in] alpha
 *          The scalar alpha
 *
 * @param[in] A
 *          The descriptor of the distributed matrix A.
 *
 * @param[in] beta
 *          The scalar beta
 *
 * @param[in,out] B
 *          The descriptor of the distributed matrix B of size M-by-N.
 *          On exit, the matrix B data are overwritten by the result of
 *          alpha * op(A) + beta * B
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_ztradd_New
 * @sa dplasma_ztradd_Destruct
 * @sa dplasma_ctradd
 * @sa dplasma_dtradd
 * @sa dplasma_stradd
 *
 ******************************************************************************/
int
dplasma_ztradd( dague_context_t *dague,
                PLASMA_enum uplo,
                PLASMA_enum trans,
                dague_complex64_t alpha,
                const tiled_matrix_desc_t *A,
                dague_complex64_t beta,
                tiled_matrix_desc_t *B)
{
    dague_handle_t *dague_ztradd = NULL;

    if ((uplo != PlasmaUpperLower) &&
        (uplo != PlasmaUpper)      &&
        (uplo != PlasmaLower))
    {
        dplasma_error("dplasma_ztradd", "illegal value of uplo");
        return -1;
    }

    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans))
    {
        dplasma_error("dplasma_ztradd", "illegal value of trans");
        return -2;
    }

    dague_ztradd = dplasma_ztradd_New(uplo, trans, alpha, A, beta, B);

    if ( dague_ztradd != NULL )
    {
        dague_enqueue(dague, (dague_handle_t*)dague_ztradd);
        dplasma_progress(dague);
        dplasma_ztradd_Destruct( dague_ztradd );
    }
    return 0;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgeadd - Generates an object that computes the operation B = alpha *
 * op(A) + beta * B, and op(A) is one of op(A) = A, or op(A) = A', or op(A) =
 * conj(A') with A and B two general matrices.
 *
 * See dplasma_tradd() for further information.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is non-transposed, transposed, or
 *          conjugate transposed
 *          = PlasmaNoTrans:   op(A) = A
 *          = PlasmaTrans:     op(A) = A'
 *          = PlasmaConjTrans: op(A) = conj(A')
 *
 * @param[in] alpha
 *          The scalar alpha
 *
 * @param[in] A
 *          The descriptor of the distributed matrix A.
 *
 * @param[in] beta
 *          The scalar beta
 *
 * @param[in,out] B
 *          The descriptor of the distributed matrix B of size M-by-N.
 *          On exit, the matrix B data are overwritten by the result of
 *          alpha * op(A) + beta * B
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeadd_New
 * @sa dplasma_zgeadd_Destruct
 * @sa dplasma_cgeadd
 * @sa dplasma_dgeadd
 * @sa dplasma_sgeadd
 *
 ******************************************************************************/
int
dplasma_zgeadd( dague_context_t *dague,
                PLASMA_enum trans,
                dague_complex64_t alpha,
                const tiled_matrix_desc_t *A,
                dague_complex64_t beta,
                tiled_matrix_desc_t *B)
{
    return dplasma_ztradd( dague, PlasmaUpperLower, trans,
                           alpha, A, beta, B );
}
