/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */
#include "dague_internal.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "dplasma/lib/memory_pool.h"

#include "zunmlq_param_LN.h"
#include "zunmlq_param_LC.h"
#include "zunmlq_param_RN.h"
#include "zunmlq_param_RC.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunmlq_param_New - Generates the dague object that overwrites the
 *  general M-by-N matrix C with
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
 *  as returned by dplasma_zgelqf_param(). Q is of order M if side = PlasmaLeft
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
 * @param[in] qrtree
 *          The structure that describes the trees used to perform the
 *          hierarchical QR factorization.
 *          See dplasma_hqr_init() or dplasma_systolic_init().
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgelqf_param_New() routine.
 *          On entry, the i-th row must contain the vector which defines the
 *          elementary reflector H(i), for i = 1,2,...,k, as returned by
 *          dplasma_zgelqf_param_New_New() in the first k rows of its array
 *          argument A.
 *          If side == PlasmaLeft,  M >= K >= 0.
 *          If side == PlasmaRight, N >= K >= 0.
 *
 * @param[in] TS
 *          Descriptor of the matrix TS distributed exactly as the A
 *          matrix. TS.mb defines the IB parameter of tile LQ algorithm. This
 *          matrix must be of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgelqf_param_New().
 *
 * @param[in] TT
 *          Descriptor of the matrix TT distributed exactly as the A
 *          matrix. TT.mb defines the IB parameter of tile LQ algorithm. This
 *          matrix must be of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgelqf_param_New().
 *
 * @param[in,out] C
 *          Descriptor of the M-by-N matrix C.
 *          On exit, the matrix C is overwritten by the result.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_zunmlq_param_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmlq_param_Destruct
 * @sa dplasma_zunmlq_param
 * @sa dplasma_cunmlq_param_New
 * @sa dplasma_dormlq_param_New
 * @sa dplasma_sormlq_param_New
 * @sa dplasma_zgelqf_param_New
 *
 ******************************************************************************/
dague_handle_t*
dplasma_zunmlq_param_New( PLASMA_enum side, PLASMA_enum trans,
                          dplasma_qrtree_t *qrtree,
                          tiled_matrix_desc_t *A,
                          tiled_matrix_desc_t *TS,
                          tiled_matrix_desc_t *TT,
                          tiled_matrix_desc_t *C)
{
    dague_handle_t* object = NULL;
    int An, ib = TS->mb;

    /* if ( !dplasma_check_desc(A) ) { */
    /*     dplasma_error("dplasma_zunmlq_param_New", "illegal A descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(T) ) { */
    /*     dplasma_error("dplasma_zunmlq_param_New", "illegal T descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(C) ) { */
    /*     dplasma_error("dplasma_zunmlq_param_New", "illegal C descriptor"); */
    /*     return NULL; */
    /* } */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zunmlq_param_New", "illegal value of side");
        return NULL;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zunmlq_param_New", "illegal value of trans");
        return NULL;
    }

    if ( side == PlasmaLeft ) {
        An = C->m;
    } else {
        An = C->n;
    }

    if ( A->m > An ) {
        dplasma_error("dplasma_zunmlq_param_New", "illegal value of A->m");
        return NULL;
    }
    if ( A->n != An ) {
        dplasma_error("dplasma_zunmlq_param_New", "illegal value of A->n");
        return NULL;
    }
    if ( (TS->nt != A->nt) || (TS->mt != A->mt) ) {
        dplasma_error("dplasma_zunmlq_param_New", "illegal size of TS (TS should have as many tiles as A)");
        return NULL;
    }
    if ( (TT->nt != A->nt) || (TT->mt != A->mt) ) {
        dplasma_error("dplasma_zunmlq_param_New", "illegal size of TT (TT should have as many tiles as A)");
        return NULL;
    }

    if ( side == PlasmaLeft ) {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_handle_t*)dague_zunmlq_param_LN_new( side, trans,
                                                                 (dague_ddesc_t*)A,
                                                                 (dague_ddesc_t*)C,
                                                                 (dague_ddesc_t*)TS,
                                                                 (dague_ddesc_t*)TT,
                                                                 *qrtree,
                                                                 NULL);
        } else {
            object = (dague_handle_t*)dague_zunmlq_param_LC_new( side, trans,
                                                                 (dague_ddesc_t*)A,
                                                                 (dague_ddesc_t*)C,
                                                                 (dague_ddesc_t*)TS,
                                                                 (dague_ddesc_t*)TT,
                                                                 *qrtree,
                                                                 NULL);
        }
    } else {
        if ( trans == PlasmaNoTrans ) {
            object = (dague_handle_t*)dague_zunmlq_param_RN_new( side, trans,
                                                                 (dague_ddesc_t*)A,
                                                                 (dague_ddesc_t*)C,
                                                                 (dague_ddesc_t*)TS,
                                                                 (dague_ddesc_t*)TT,
                                                                 *qrtree,
                                                                 NULL);
        } else {
            object = (dague_handle_t*)dague_zunmlq_param_RC_new( side, trans,
                                                                 (dague_ddesc_t*)A,
                                                                 (dague_ddesc_t*)C,
                                                                 (dague_ddesc_t*)TS,
                                                                 (dague_ddesc_t*)TT,
                                                                 *qrtree,
                                                                 NULL);
        }
    }

    ((dague_zunmlq_param_LC_handle_t*)object)->p_work = (dague_memory_pool_t*)malloc(sizeof(dague_memory_pool_t));
    dague_private_memory_init( ((dague_zunmlq_param_LC_handle_t*)object)->p_work, ib * TS->nb * sizeof(dague_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( ((dague_zunmlq_param_LC_handle_t*)object)->arenas[DAGUE_zunmlq_param_LC_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( ((dague_zunmlq_param_LC_handle_t*)object)->arenas[DAGUE_zunmlq_param_LC_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 0 );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( ((dague_zunmlq_param_LC_handle_t*)object)->arenas[DAGUE_zunmlq_param_LC_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(dague_complex64_t),
                             DAGUE_ARENA_ALIGNMENT_SSE,
                             MPI_DOUBLE_COMPLEX, A->mb, 1 );

    /* Little T */
    dplasma_add2arena_rectangle( ((dague_zunmlq_param_LC_handle_t*)object)->arenas[DAGUE_zunmlq_param_LC_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, TS->mb, TS->nb, -1);

    return object;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zunmlq_param_Destruct - Free the data structure associated to an object
 *  created with dplasma_zunmlq_param_New().
 *
 *******************************************************************************
 *
 * @param[in,out] object
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmlq_param_New
 * @sa dplasma_zunmlq_param
 *
 ******************************************************************************/
void
dplasma_zunmlq_param_Destruct( dague_handle_t *object )
{
    dague_zunmlq_param_LC_handle_t *dague_zunmlq_param = (dague_zunmlq_param_LC_handle_t *)object;

    dplasma_datatype_undefine_type( &(dague_zunmlq_param->arenas[DAGUE_zunmlq_param_LC_LOWER_TILE_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zunmlq_param->arenas[DAGUE_zunmlq_param_LC_LITTLE_T_ARENA  ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zunmlq_param->arenas[DAGUE_zunmlq_param_LC_DEFAULT_ARENA   ]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(dague_zunmlq_param->arenas[DAGUE_zunmlq_param_LC_UPPER_TILE_ARENA]->opaque_dtt) );

    dague_private_memory_fini( dague_zunmlq_param->p_work );
    free( dague_zunmlq_param->p_work );

    DAGUE_INTERNAL_HANDLE_DESTRUCT(dague_zunmlq_param);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_Complex64_t
 *
 *  dplasma_zunmlq_param - Generates the dague object that overwrites the general
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
 *  as returned by dplasma_zgelqf_param(). Q is of order M if side = PlasmaLeft
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
 * @param[in] qrtree
 *          The structure that describes the trees used to perform the
 *          hierarchical QR factorization.
 *          See dplasma_hqr_init() or dplasma_systolic_init().
 *
 * @param[in] qrtree
 *          The structure that describes the trees used to perform the
 *          hierarchical QR factorization.
 *          See dplasma_hqr_init() or dplasma_systolic_init().
 *
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K factorized with the
 *          dplasma_zgelqf_New() routine.
 *          On entry, the i-th row must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgelqf_New() in the first k rows of its array
 *          argument A.
 *          If side == PlasmaLeft,  M >= K >= 0.
 *          If side == PlasmaRight, N >= K >= 0.
 *
 * @param[in] TS
 *          Descriptor of the matrix TS distributed exactly as the A
 *          matrix. TS.mb defines the IB parameter of tile LQ algorithm. This
 *          matrix must be of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgelqf_param_New().
 *
 * @param[in] TT
 *          Descriptor of the matrix TT distributed exactly as the A
 *          matrix. TT.mb defines the IB parameter of tile LQ algorithm. This
 *          matrix must be of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgelqf_param_New().
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
 * @sa dplasma_zunmlq_param_New
 * @sa dplasma_zunmlq_param_Destruct
 * @sa dplasma_cunmlq_param
 * @sa dplasma_dormlq_param
 * @sa dplasma_sormlq_param
 * @sa dplasma_zgelqf_param
 *
 ******************************************************************************/
int
dplasma_zunmlq_param( dague_context_t *dague,
                      PLASMA_enum side, PLASMA_enum trans,
                      dplasma_qrtree_t    *qrtree,
                      tiled_matrix_desc_t *A,
                      tiled_matrix_desc_t *TS,
                      tiled_matrix_desc_t *TT,
                      tiled_matrix_desc_t *C )
{
    dague_handle_t *dague_zunmlq_param = NULL;
    int An;

    if (dague == NULL) {
        dplasma_error("dplasma_zunmlq_param", "dplasma not initialized");
        return -1;
    }

    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zunmlq_param", "illegal value of side");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zunmlq_param", "illegal value of trans");
        return -2;
    }

    if ( side == PlasmaLeft ) {
        An = C->m;
    } else {
        An = C->n;
    }
    if ( A->n != An ) {
        dplasma_error("dplasma_zunmlq_param", "illegal value of A->n");
        return -3;
    }
    if ( A->m > An ) {
        dplasma_error("dplasma_zunmlq_param", "illegal value of A->m");
        return -5;
    }
    if ( (TS->nt != A->nt) || (TS->mt != A->mt) ) {
        dplasma_error("dplasma_zunmlq_param", "illegal size of TS (TS should have as many tiles as A)");
        return -20;
    }
    if ( (TT->nt != A->nt) || (TT->mt != A->mt) ) {
        dplasma_error("dplasma_zunmlq_param", "illegal size of TT (TT should have as many tiles as A)");
        return -20;
    }

    if (dplasma_imin(C->m, dplasma_imin(C->n, A->m)) == 0)
        return 0;

    dague_zunmlq_param = dplasma_zunmlq_param_New(side, trans, qrtree, A, TS, TT, C);

    if ( dague_zunmlq_param != NULL ){
        dague_enqueue(dague, (dague_handle_t*)dague_zunmlq_param);
        dplasma_progress(dague);
        dplasma_zunmlq_param_Destruct( dague_zunmlq_param );
    }

    return 0;
}
