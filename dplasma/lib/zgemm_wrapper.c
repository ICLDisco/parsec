/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zgemm_NN.h"
#include "zgemm_NT.h"
#include "zgemm_TN.h"
#include "zgemm_TT.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgemm_New - Generates the object that performs one of the following
 *  matrix-matrix operations. WARNING: The computations are not done by this call.
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = conjg( X' )
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or conjugate transposed:
 *          = PlasmaNoTrans:   A is not transposed;
 *          = PlasmaTrans:     A is transposed;
 *          = PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or conjugate transposed:
 *          = PlasmaNoTrans:   B is not transposed;
 *          = PlasmaTrans:     B is transposed;
 *          = PlasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C are overwritten by the matrix (
 *          alpha*op( A )*op( B ) + beta*C )
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgemm_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm
 * @sa dplasma_zgemm_Destruct
 * @sa dplasma_cgemm_New
 * @sa dplasma_dgemm_New
 * @sa dplasma_sgemm_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zgemm_New( PLASMA_enum transA, PLASMA_enum transB,
                   dague_complex64_t alpha, const tiled_matrix_desc_t* A, const tiled_matrix_desc_t* B,
                   dague_complex64_t beta,  tiled_matrix_desc_t* C)
{
    dague_handle_t* zgemm_object;
    dague_arena_t* arena;

    /* Check input arguments */
    if ((transA != PlasmaNoTrans) && (transA != PlasmaTrans) && (transA != PlasmaConjTrans)) {
        dplasma_error("dplasma_zgemm_New", "illegal value of transA");
        return NULL /*-1*/;
    }
    if ((transB != PlasmaNoTrans) && (transB != PlasmaTrans) && (transB != PlasmaConjTrans)) {
        dplasma_error("dplasma_zgemm_New", "illegal value of transB");
        return NULL /*-2*/;
    }

    if( PlasmaNoTrans == transA ) {
        if( PlasmaNoTrans == transB ) {
            two_dim_block_cyclic_t *Adist, *Bdist;
            int P, Q;

            /* Create two fake descriptors for task distribution of the read tasks */
            {
                P = ((two_dim_block_cyclic_t*)A)->grid.rows;
                Q = ((two_dim_block_cyclic_t*)A)->grid.cols;

                Adist = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));

                two_dim_block_cyclic_init(
                    Adist, matrix_RealDouble, matrix_Tile,
                    A->super.nodes, A->super.myrank,
                    1, 1, /* Dimensions of the tiles              */
                    dplasma_imax(A->mt, P), Q, /* Dimensions of the matrix             */
                    0, 0, /* Starting points (not important here) */
                    dplasma_imax(A->mt, P), Q, /* Dimensions of the sub-matrix         */
                    1, 1, P);
                Adist->super.super.data_of = fake_data_of;
            }

            {
                P = ((two_dim_block_cyclic_t*)B)->grid.rows;
                Q = ((two_dim_block_cyclic_t*)B)->grid.cols;

                Bdist = (two_dim_block_cyclic_t*)malloc(sizeof(two_dim_block_cyclic_t));

                two_dim_block_cyclic_init(
                    Bdist, matrix_RealDouble, matrix_Tile,
                    B->super.nodes, B->super.myrank,
                    1, 1, /* Dimensions of the tiles              */
                    P, dplasma_imax(B->nt, Q), /* Dimensions of the matrix             */
                    0, 0, /* Starting points (not important here) */
                    P, dplasma_imax(B->nt, Q), /* Dimensions of the sub-matrix         */
                    1, 1, P);
                Bdist->super.super.data_of = fake_data_of;
            }

            dague_zgemm_NN_handle_t* object;
            object = dague_zgemm_NN_new(transA, transB, alpha, beta,
                                        (dague_ddesc_t*)A, (dague_ddesc_t*)Adist,
                                        (dague_ddesc_t*)B, (dague_ddesc_t*)Bdist,
                                        (dague_ddesc_t*)C,
                                        ((two_dim_block_cyclic_t*)B)->grid.rows,
                                        ((two_dim_block_cyclic_t*)A)->grid.cols);
            arena = object->arenas[DAGUE_zgemm_NN_DEFAULT_ARENA];
            zgemm_object = (dague_handle_t*)object;
        } else {
            dague_zgemm_NT_handle_t* object;
            object = dague_zgemm_NT_new(transA, transB, alpha, beta,
                                        (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)B,
                                        (dague_ddesc_t*)C);
            arena = object->arenas[DAGUE_zgemm_NT_DEFAULT_ARENA];
            zgemm_object = (dague_handle_t*)object;
        }
    } else {
        if( PlasmaNoTrans == transB ) {
            dague_zgemm_TN_handle_t* object;
            object = dague_zgemm_TN_new(transA, transB, alpha, beta,
                                        (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)B,
                                        (dague_ddesc_t*)C);
            arena = object->arenas[DAGUE_zgemm_TN_DEFAULT_ARENA];
            zgemm_object = (dague_handle_t*)object;
        } else {
            dague_zgemm_TT_handle_t* object;
            object = dague_zgemm_TT_new(transA, transB, alpha, beta,
                                        (dague_ddesc_t*)A,
                                        (dague_ddesc_t*)B,
                                        (dague_ddesc_t*)C);
            arena = object->arenas[DAGUE_zgemm_TT_DEFAULT_ARENA];
            zgemm_object = (dague_handle_t*)object;
        }
    }

    dplasma_add2arena_tile(arena,
                           A->mb*A->nb*sizeof(dague_complex64_t),
                           DAGUE_ARENA_ALIGNMENT_SSE,
                           MPI_DOUBLE_COMPLEX, A->mb);

    return zgemm_object;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgemm_Destruct - Free the data structure associated to an object
 *  created with dplasma_zgemm_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm_New
 * @sa dplasma_zgemm
 *
 ******************************************************************************/
void
dplasma_zgemm_Destruct( dague_handle_t *o )
{
    //dague_matrix_del2arena( ((dague_zgemm_NN_handle_t *)o)->arenas[DAGUE_zgemm_NN_DEFAULT_ARENA] );

    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgemm - Performs one of the following matrix-matrix operations
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = conjg( X' )
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or conjugate transposed:
 *          = PlasmaNoTrans:   A is not transposed;
 *          = PlasmaTrans:     A is transposed;
 *          = PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or conjugate transposed:
 *          = PlasmaNoTrans:   B is not transposed;
 *          = PlasmaTrans:     B is transposed;
 *          = PlasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C are overwritten by the matrix (
 *          alpha*op( A )*op( B ) + beta*C )
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm_New
 * @sa dplasma_zgemm_Destruct
 * @sa dplasma_cgemm
 * @sa dplasma_dgemm
 * @sa dplasma_sgemm
 *
 ******************************************************************************/
int
dplasma_zgemm( dague_context_t *dague,
               PLASMA_enum transA, PLASMA_enum transB,
               dague_complex64_t alpha, const tiled_matrix_desc_t *A,
                                        const tiled_matrix_desc_t *B,
               dague_complex64_t beta,        tiled_matrix_desc_t *C)
{
    dague_handle_t *dague_zgemm = NULL;
    int M, N, K;
    int Am, An, Ai, Aj, Amb, Anb;
    int Bm, Bn, Bi, Bj, Bmb, Bnb;

    /* Check input arguments */
    if ((transA != PlasmaNoTrans) && (transA != PlasmaTrans) && (transA != PlasmaConjTrans)) {
        dplasma_error("dplasma_zgemm", "illegal value of transA");
        return -1;
    }
    if ((transB != PlasmaNoTrans) && (transB != PlasmaTrans) && (transB != PlasmaConjTrans)) {
        dplasma_error("dplasma_zgemm", "illegal value of transB");
        return -2;
    }

    if ( transA == PlasmaNoTrans ) {
        Am  = A->m;
        An  = A->n;
        Amb = A->mb;
        Anb = A->nb;
        Ai  = A->i;
        Aj  = A->j;
    } else {
        Am  = A->n;
        An  = A->m;
        Amb = A->nb;
        Anb = A->mb;
        Ai  = A->j;
        Aj  = A->i;
    }

    if ( transB == PlasmaNoTrans ) {
        Bm  = B->m;
        Bn  = B->n;
        Bmb = B->mb;
        Bnb = B->nb;
        Bi  = B->i;
        Bj  = B->j;
    } else {
        Bm  = B->n;
        Bn  = B->m;
        Bmb = B->nb;
        Bnb = B->mb;
        Bi  = B->j;
        Bj  = B->i;
    }

    if ( (Amb != C->mb) || (Anb != Bmb) || (Bnb != C->nb) ) {
        dplasma_error("dplasma_zgemm", "tile sizes have to match");
        return -101;
    }
    if ( (Am != C->m) || (An != Bm) || (Bn != C->n) ) {
        dplasma_error("dplasma_zgemm", "sizes of matrices have to match");
        return -101;
    }
    if ( (Ai != C->i) || (Aj != Bi) || (Bj != C->j) ) {
        dplasma_error("dplasma_zgemm", "start indexes have to match");
        return -101;
    }

    M = C->m;
    N = C->n;
    K = An;

    /* Quick return */
    if (M == 0 || N == 0 ||
        ((alpha == (PLASMA_Complex64_t)0.0 || K == 0) && beta == (PLASMA_Complex64_t)1.0))
        return 0;

    dague_zgemm = dplasma_zgemm_New(transA, transB,
                                    alpha, A, B,
                                    beta, C);

    if ( dague_zgemm != NULL )
    {
        dague_enqueue( dague, (dague_handle_t*)dague_zgemm);
        dplasma_progress(dague);
        dplasma_zgemm_Destruct( dague_zgemm );
        return 0;
    }
    else {
        return -101;
    }
}
