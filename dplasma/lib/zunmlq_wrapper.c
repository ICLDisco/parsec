/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
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
#include "dplasma/lib/dplasmaaux.h"
#include "dague/private_mempool.h"

#include "zunmlq_LN.h"
#include "zunmlq_LC.h"
#include "zunmlq_RN.h"
#include "zunmlq_RC.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunmlq_New - Generates the dague handle that overwrites the general
 *  M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'C':      Q**H * C       C * Q**H
 *
 *  where Q is a unitary matrix defined as the product of k elementary
 *  reflectors
 *
 *        Q = H(1) H(2) . . . H(k)
 *
 *  as returned by dplasma_zgelqf(). Q is of order M if side = PlasmaLeft
 *  and of order N if side = PlasmaRight.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          @arg PlasmaLeft:  apply Q or Q**H from the left;
 *          @arg PlasmaRight: apply Q or Q**H from the right.
 *
 * @param[in] trans
 *          @arg PlasmaNoTrans:   no transpose, apply Q;
 *          @arg PlasmaConjTrans: conjugate transpose, apply Q**H.
 *
 * @param[in] A
 *          Descriptor of the matrix A of size K-by-M factorized with the
 *          dplasma_zgelqf_New() routine.
 *          On entry, the i-th row must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgelqf_New() in the first k rows of its array
 *          argument A.
 *          If side == PlasmaLeft,  M >= K >= 0.
 *          If side == PlasmaRight, N >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile LQ algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgelqf_New().
 *
 * @param[in,out] C
 *          Descriptor of the M-by-N matrix C.
 *          On exit, the matrix C is overwritten by the result.
 *
 *******************************************************************************
 *
 * @return
 *          \retval The dague handle which describes the operation to perform
 *                  NULL if one of the parameter is incorrect
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmlq_Destruct
 * @sa dplasma_zunmlq
 * @sa dplasma_cunmlq_New
 * @sa dplasma_dormlq_New
 * @sa dplasma_sormlq_New
 * @sa dplasma_zgelqf_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zunmlq_New( PLASMA_enum side, PLASMA_enum trans,
                    tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *T,
                    tiled_matrix_desc_t *C )
{
    dague_handle_t* handle;
    int An, ib = T->mb;

    /* if ( !dplasma_check_desc(A) ) { */
    /*     dplasma_error("dplasma_zunmlq_New", "illegal A descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(T) ) { */
    /*     dplasma_error("dplasma_zunmlq_New", "illegal T descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(C) ) { */
    /*     dplasma_error("dplasma_zunmlq_New", "illegal C descriptor"); */
    /*     return NULL; */
    /* } */
    if ( side == PlasmaLeft ) {
        An = C->m;
    } else {
        An = C->n;
    }
    if ( A->n != An ) {
        dplasma_error("dplasma_zunmlq_New", "illegal value of A->n");
        return NULL;
    }
    if ( A->m > An ) {
        dplasma_error("dplasma_zunmlq_New", "illegal value of A->m");
        return NULL;
    }
    if ( (T->nt != A->nt) || (T->mt != A->mt) ) {
        dplasma_error("dplasma_zunmlq_New", "illegal size of T (T should have as many tiles as A)");
        return NULL;
    }

    if ( side == PlasmaLeft ) {
        if ( trans == PlasmaNoTrans ) {
            handle = (dague_handle_t*)dague_zunmlq_LN_new( side, trans,
                                                           (dague_ddesc_t*)A,
                                                           (dague_ddesc_t*)C,
                                                           (dague_ddesc_t*)T,
                                                           NULL);
        } else {
            handle = (dague_handle_t*)dague_zunmlq_LC_new( side, trans,
                                                           (dague_ddesc_t*)A,
                                                           (dague_ddesc_t*)C,
                                                           (dague_ddesc_t*)T,
                                                           NULL);
        }
    } else {
        if ( trans == PlasmaNoTrans ) {
            handle = (dague_handle_t*)dague_zunmlq_RN_new( side, trans,
                                                           (dague_ddesc_t*)A,
                                                           (dague_ddesc_t*)C,
                                                           (dague_ddesc_t*)T,
                                                           NULL);
        } else {
            handle = (dague_handle_t*)dague_zunmlq_RC_new( side, trans,
                                                           (dague_ddesc_t*)A,
                                                           (dague_ddesc_t*)C,
                                                           (dague_ddesc_t*)T,
                                                           NULL);
        }
    }

    ((dague_zunmlq_LC_handle_t*)handle)->pool_0 = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( ((dague_zunmlq_LC_handle_t*)handle)->pool_0, ib * T->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( ((dague_zunmlq_LC_handle_t*)handle)->arenas[DAGUE_zunmlq_LC_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            dague_datatype_double_complex_t, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_upper( ((dague_zunmlq_LC_handle_t*)handle)->arenas[DAGUE_zunmlq_LC_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             dague_datatype_double_complex_t, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( ((dague_zunmlq_LC_handle_t*)handle)->arenas[DAGUE_zunmlq_LC_LITTLE_T_ARENA],
                                 T->mb*T->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 dague_datatype_double_complex_t, T->mb, T->nb, -1);

    return handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunmlq_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zunmlq_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmlq_New
 * @sa dplasma_zunmlq
 *
 ******************************************************************************/
void
dplasma_zunmlq_Destruct( dague_handle_t *handle )
{
    dague_zunmlq_LC_handle_t *dague_zunmlq = (dague_zunmlq_LC_handle_t *)handle;

    dague_matrix_del2arena( dague_zunmlq->arenas[DAGUE_zunmlq_LC_DEFAULT_ARENA   ] );
    dague_matrix_del2arena( dague_zunmlq->arenas[DAGUE_zunmlq_LC_UPPER_TILE_ARENA] );
    dague_matrix_del2arena( dague_zunmlq->arenas[DAGUE_zunmlq_LC_LITTLE_T_ARENA  ] );

    dague_private_memory_fini( dague_zunmlq->pool_0 );
    free( dague_zunmlq->pool_0 );

    handle->destructor(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunmlq_New - Overwrites the general M-by-N matrix C with
 *
 *                  SIDE = 'L'     SIDE = 'R'
 *  TRANS = 'N':      Q * C          C * Q
 *  TRANS = 'C':      Q**H * C       C * Q**H
 *
 *  where Q is a unitary matrix defined as the product of k elementary
 *  reflectors
 *
 *        Q = H(1) H(2) . . . H(k)
 *
 *  as returned by dplasma_zgelqf(). Q is of order M if side = PlasmaLeft
 *  and of order N if side = PlasmaRight.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] side
 *          @arg PlasmaLeft:  apply Q or Q**H from the left;
 *          @arg PlasmaRight: apply Q or Q**H from the right.
 *
 * @param[in] trans
 *          @arg PlasmaNoTrans:   no transpose, apply Q;
 *          @arg PlasmaConjTrans: conjugate transpose, apply Q**H.
 *
 * @param[in] A
 *          Descriptor of the matrix A of size K-by-(N,M) factorized with the
 *          dplasma_zgelqf_New() routine.
 *          On entry, the i-th row must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgelqf_New() in the first k rows of its array
 *          argument A.
 *          If side == PlasmaLeft,  M >= K >= 0.
 *          If side == PlasmaRight, N >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile LQ algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgelqf_New().
 *
 * @param[in,out] C
 *          Descriptor of the M-by-N matrix C.
 *          On exit, the matrix C is overwritten by the result.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmlq_Destruct
 * @sa dplasma_zunmlq
 * @sa dplasma_cunmlq_New
 * @sa dplasma_dormlq_New
 * @sa dplasma_sormlq_New
 * @sa dplasma_zgelqf_New
 *
 ******************************************************************************/
int
dplasma_zunmlq( dague_context_t *dague,
                PLASMA_enum side, PLASMA_enum trans,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *T,
                tiled_matrix_desc_t *C )
{
    dague_handle_t *dague_zunmlq = NULL;
    int An;

    if (dague == NULL) {
        dplasma_error("dplasma_zunmlq", "dplasma not initialized");
        return -1;
    }

    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zunmlq_New", "illegal value of side");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zunmlq_New", "illegal value of trans");
        return -2;
    }

    if ( side == PlasmaLeft ) {
        An = C->m;
    } else {
        An = C->n;
    }
    if ( A->n != An ) {
        dplasma_error("dplasma_zunmlq_New", "illegal value of A->n");
        return -3;
    }
    if ( A->m > An ) {
        dplasma_error("dplasma_zunmlq_New", "illegal value of A->m");
        return -5;
    }
    if ( (T->nt != A->nt) || (T->mt != A->mt) ) {
        dplasma_error("dplasma_zunmlq_New", "illegal size of T (T should have as many tiles as A)");
        return -20;
    }

    if (dplasma_imin(C->m, dplasma_imin(C->n, A->m)) == 0)
        return 0;

    dague_zunmlq = dplasma_zunmlq_New(side, trans, A, T, C);

    if ( dague_zunmlq != NULL ){
        dague_enqueue(dague, (dague_handle_t*)dague_zunmlq);
        dplasma_progress(dague);
        dplasma_zunmlq_Destruct( dague_zunmlq );
    }
    return 0;
}
