/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "parsec/private_mempool.h"

#include "zunmqr_param_LN.h"
#include "zunmqr_param_LC.h"
#include "zunmqr_param_RN.h"
#include "zunmqr_param_RC.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunmqr_param_New - Generates the parsec handle that overwrites the
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
 *  as returned by dplasma_zgeqrf_param(). Q is of order M if side = PlasmaLeft
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
 *          dplasma_zgeqrf_param_New() routine.
 *          On entry, the i-th column must contain the vector which defines the
 *          elementary reflector H(i), for i = 1,2,...,k, as returned by
 *          dplasma_zgeqrf_param_New_New() in the first k columns of its array
 *          argument A.
 *          If side == PlasmaLeft,  M >= K >= 0.
 *          If side == PlasmaRight, N >= K >= 0.
 *
 * @param[in] TS
 *          Descriptor of the matrix TS distributed exactly as the A
 *          matrix. TS.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
 *
 * @param[in] TT
 *          Descriptor of the matrix TT distributed exactly as the A
 *          matrix. TT.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
 *
 * @param[in,out] C
 *          Descriptor of the M-by-N matrix C.
 *          On exit, the matrix C is overwritten by the result.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec handle describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zunmqr_param_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmqr_param_Destruct
 * @sa dplasma_zunmqr_param
 * @sa dplasma_cunmqr_param_New
 * @sa dplasma_dormqr_param_New
 * @sa dplasma_sormqr_param_New
 * @sa dplasma_zgeqrf_param_New
 *
 ******************************************************************************/
parsec_handle_t*
dplasma_zunmqr_param_New( PLASMA_enum side, PLASMA_enum trans,
                          dplasma_qrtree_t *qrtree,
                          tiled_matrix_desc_t *A,
                          tiled_matrix_desc_t *TS,
                          tiled_matrix_desc_t *TT,
                          tiled_matrix_desc_t *C)
{
    parsec_handle_t* handle = NULL;
    int Am, ib = TS->mb;

    /* if ( !dplasma_check_desc(A) ) { */
    /*     dplasma_error("dplasma_zunmqr_param_New", "illegal A descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(T) ) { */
    /*     dplasma_error("dplasma_zunmqr_param_New", "illegal T descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(C) ) { */
    /*     dplasma_error("dplasma_zunmqr_param_New", "illegal C descriptor"); */
    /*     return NULL; */
    /* } */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal value of side");
        return NULL;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal value of trans");
        return NULL;
    }

    if ( side == PlasmaLeft ) {
        Am = C->m;
    } else {
        Am = C->n;
    }

    if ( A->n > Am ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal value of A->n");
        return NULL;
    }
    if ( A->m != Am ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal value of A->m");
        return NULL;
    }
    if ( (TS->nt != A->nt) || (TS->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal size of TS (TS should have as many tiles as A)");
        return NULL;
    }
    if ( (TT->nt != A->nt) || (TT->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr_param_New", "illegal size of TT (TT should have as many tiles as A)");
        return NULL;
    }

    if ( side == PlasmaLeft ) {
        if ( trans == PlasmaNoTrans ) {
            handle = (parsec_handle_t*)parsec_zunmqr_param_LN_new( side, trans,
                                                                 A,
                                                                 C,
                                                                 TS,
                                                                 TT,
                                                                 *qrtree,
                                                                 NULL);
        } else {
            handle = (parsec_handle_t*)parsec_zunmqr_param_LC_new( side, trans,
                                                                 A,
                                                                 C,
                                                                 TS,
                                                                 TT,
                                                                 *qrtree,
                                                                 NULL);
        }
    } else {
        if ( trans == PlasmaNoTrans ) {
            handle = (parsec_handle_t*)parsec_zunmqr_param_RN_new( side, trans,
                                                                 A,
                                                                 C,
                                                                 TS,
                                                                 TT,
                                                                 *qrtree,
                                                                 NULL);
        } else {
            handle = (parsec_handle_t*)parsec_zunmqr_param_RC_new( side, trans,
                                                                 A,
                                                                 C,
                                                                 TS,
                                                                 TT,
                                                                 *qrtree,
                                                                 NULL);
        }
    }

    ((parsec_zunmqr_param_LC_handle_t*)handle)->_g_p_work = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( ((parsec_zunmqr_param_LC_handle_t*)handle)->_g_p_work, ib * TS->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( ((parsec_zunmqr_param_LC_handle_t*)handle)->arenas[PARSEC_zunmqr_param_LC_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( ((parsec_zunmqr_param_LC_handle_t*)handle)->arenas[PARSEC_zunmqr_param_LC_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* Upper triangular part of tile with diagonal */
    dplasma_add2arena_upper( ((parsec_zunmqr_param_LC_handle_t*)handle)->arenas[PARSEC_zunmqr_param_LC_UPPER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 1 );

    /* Little T */
    dplasma_add2arena_rectangle( ((parsec_zunmqr_param_LC_handle_t*)handle)->arenas[PARSEC_zunmqr_param_LC_LITTLE_T_ARENA],
                                 TS->mb*TS->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, TS->mb, TS->nb, -1);

    return handle;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunmqr_param_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zunmqr_param_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmqr_param_New
 * @sa dplasma_zunmqr_param
 *
 ******************************************************************************/
void
dplasma_zunmqr_param_Destruct( parsec_handle_t *handle )
{
    parsec_zunmqr_param_LC_handle_t *parsec_zunmqr_param = (parsec_zunmqr_param_LC_handle_t *)handle;

    parsec_matrix_del2arena( parsec_zunmqr_param->arenas[PARSEC_zunmqr_param_LC_LOWER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zunmqr_param->arenas[PARSEC_zunmqr_param_LC_LITTLE_T_ARENA  ] );
    parsec_matrix_del2arena( parsec_zunmqr_param->arenas[PARSEC_zunmqr_param_LC_DEFAULT_ARENA   ] );
    parsec_matrix_del2arena( parsec_zunmqr_param->arenas[PARSEC_zunmqr_param_LC_UPPER_TILE_ARENA] );

    parsec_private_memory_fini( parsec_zunmqr_param->_g_p_work );
    free( parsec_zunmqr_param->_g_p_work );

    parsec_handle_free(handle);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunmqr_param - Generates the parsec handle that overwrites the general
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
 *  as returned by dplasma_zgeqrf_param(). Q is of order M if side = PlasmaLeft
 *  and of order N if side = PlasmaRight.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
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
 *          dplasma_zgeqrf_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_New() in the first k columns of its array
 *          argument A.
 *          If side == PlasmaLeft,  M >= K >= 0.
 *          If side == PlasmaRight, N >= K >= 0.
 *
 * @param[in] TS
 *          Descriptor of the matrix TS distributed exactly as the A
 *          matrix. TS.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TS.mb - by - A.nt * TS.nb, with TS.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
 *
 * @param[in] TT
 *          Descriptor of the matrix TT distributed exactly as the A
 *          matrix. TT.mb defines the IB parameter of tile QR algorithm. This
 *          matrix must be of size A.mt * TT.mb - by - A.nt * TT.nb, with TT.nb
 *          == A.nb.  This matrix is initialized during the call to
 *          dplasma_zgeqrf_param_New().
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
 * @sa dplasma_zunmqr_param_New
 * @sa dplasma_zunmqr_param_Destruct
 * @sa dplasma_cunmqr_param
 * @sa dplasma_dormqr_param
 * @sa dplasma_sormqr_param
 * @sa dplasma_zgeqrf_param
 *
 ******************************************************************************/
int
dplasma_zunmqr_param( parsec_context_t *parsec,
                      PLASMA_enum side, PLASMA_enum trans,
                      dplasma_qrtree_t    *qrtree,
                      tiled_matrix_desc_t *A,
                      tiled_matrix_desc_t *TS,
                      tiled_matrix_desc_t *TT,
                      tiled_matrix_desc_t *C )
{
    parsec_handle_t *parsec_zunmqr_param = NULL;
    int Am;

    if (parsec == NULL) {
        dplasma_error("dplasma_zunmqr_param", "dplasma not initialized");
        return -1;
    }

    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zunmqr_param", "illegal value of side");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zunmqr_param", "illegal value of trans");
        return -2;
    }

    if ( side == PlasmaLeft ) {
        Am = C->m;
    } else {
        Am = C->n;
    }
    if ( A->m != Am ) {
        dplasma_error("dplasma_zunmqr_param", "illegal value of A->m");
        return -3;
    }
    if ( A->n > Am ) {
        dplasma_error("dplasma_zunmqr_param", "illegal value of A->n");
        return -5;
    }
    if ( (TS->nt != A->nt) || (TS->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr_param", "illegal size of TS (TS should have as many tiles as A)");
        return -20;
    }
    if ( (TT->nt != A->nt) || (TT->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr_param", "illegal size of TT (TT should have as many tiles as A)");
        return -20;
    }

    if (dplasma_imin(C->m, dplasma_imin(C->n, A->n)) == 0)
        return 0;

    parsec_zunmqr_param = dplasma_zunmqr_param_New(side, trans, qrtree, A, TS, TT, C);

    if ( parsec_zunmqr_param != NULL ){
        parsec_enqueue(parsec, (parsec_handle_t*)parsec_zunmqr_param);
        dplasma_progress(parsec);
        dplasma_zunmqr_param_Destruct( parsec_zunmqr_param );
    }

    return 0;
}
