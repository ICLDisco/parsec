/*
 * Copyright (c) 2011-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"
#include "dplasma/lib/dplasmaaux.h"
#include "parsec/private_mempool.h"

#include "zunmqr_LN.h"
#include "zunmqr_LC.h"
#include "zunmqr_RN.h"
#include "zunmqr_RC.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunmqr_New - Generates the parsec handle that overwrites the general
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
 *  as returned by dplasma_zgeqrf(). Q is of order M if side = PlasmaLeft
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
 *          Descriptor of the matrix A of size M-by-K if side == PlasmaLeft, or
 *          N-by-K if side == PlasmaRight factorized with the
 *          dplasma_zgeqrf_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_New() in the first k columns of its array
 *          argument A.
 *          If side == PlasmaLeft,  M >= K >= 0.
 *          If side == PlasmaRight, N >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgeqrf_New().
 *
 * @param[in,out] C
 *          Descriptor of the M-by-N matrix C.
 *          On exit, the matrix C is overwritten by the result.
 *
 *******************************************************************************
 *
 * @return
 *          \retval The parsec handle which describes the operation to perform
 *                  NULL if one of the parameter is incorrect
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmqr_Destruct
 * @sa dplasma_zunmqr
 * @sa dplasma_cunmqr_New
 * @sa dplasma_dormqr_New
 * @sa dplasma_sormqr_New
 * @sa dplasma_zgeqrf_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zunmqr_New( PLASMA_enum side, PLASMA_enum trans,
                    tiled_matrix_desc_t *A,
                    tiled_matrix_desc_t *T,
                    tiled_matrix_desc_t *C )
{
    parsec_taskpool_t* tp;
    int Am, ib = T->mb;

    /* if ( !dplasma_check_desc(A) ) { */
    /*     dplasma_error("dplasma_zunmqr_New", "illegal A descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(T) ) { */
    /*     dplasma_error("dplasma_zunmqr_New", "illegal T descriptor"); */
    /*     return NULL; */
    /* } */
    /* if ( !dplasma_check_desc(C) ) { */
    /*     dplasma_error("dplasma_zunmqr_New", "illegal C descriptor"); */
    /*     return NULL; */
    /* } */
    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of side");
        return NULL;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of trans");
        return NULL;
    }

    if ( side == PlasmaLeft ) {
        Am = C->m;
    } else {
        Am = C->n;
    }

    if ( A->n > Am ) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of A->n");
        return NULL;
    }
    if ( A->m != Am ) {
        dplasma_error("dplasma_zunmqr_New", "illegal value of A->m");
        return NULL;
    }
    if ( (T->nt != A->nt) || (T->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr_New", "illegal size of T (T should have as many tiles as A)");
        return NULL;
    }

    if ( side == PlasmaLeft ) {
        if ( trans == PlasmaNoTrans ) {
            tp = (parsec_taskpool_t*)parsec_zunmqr_LN_new( side, trans,
                                                           A,
                                                           C,
                                                           T,
                                                           NULL);
        } else {
            tp = (parsec_taskpool_t*)parsec_zunmqr_LC_new( side, trans,
                                                           A,
                                                           C,
                                                           T,
                                                           NULL);
        }
    } else {
        if ( trans == PlasmaNoTrans ) {
            tp = (parsec_taskpool_t*)parsec_zunmqr_RN_new( side, trans,
                                                           A,
                                                           C,
                                                           T,
                                                           NULL);
        } else {
            tp = (parsec_taskpool_t*)parsec_zunmqr_RC_new( side, trans,
                                                           A,
                                                           C,
                                                           T,
                                                           NULL);
        }
    }

    ((parsec_zunmqr_LC_taskpool_t*)tp)->_g_pool_0 = (parsec_memory_pool_t*)malloc(sizeof(parsec_memory_pool_t));
    parsec_private_memory_init( ((parsec_zunmqr_LC_taskpool_t*)tp)->_g_pool_0, ib * T->nb * sizeof(parsec_complex64_t) );

    /* Default type */
    dplasma_add2arena_tile( ((parsec_zunmqr_LC_taskpool_t*)tp)->arenas[PARSEC_zunmqr_LC_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb );

    /* Lower triangular part of tile without diagonal */
    dplasma_add2arena_lower( ((parsec_zunmqr_LC_taskpool_t*)tp)->arenas[PARSEC_zunmqr_LC_LOWER_TILE_ARENA],
                             A->mb*A->nb*sizeof(parsec_complex64_t),
                             PARSEC_ARENA_ALIGNMENT_SSE,
                             parsec_datatype_double_complex_t, A->mb, 0 );

    /* Little T */
    dplasma_add2arena_rectangle( ((parsec_zunmqr_LC_taskpool_t*)tp)->arenas[PARSEC_zunmqr_LC_LITTLE_T_ARENA],
                                 T->mb*T->nb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, T->mb, T->nb, -1);

    return tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunmqr_Destruct - Free the data structure associated to an handle
 *  created with dplasma_zunmqr_New().
 *
 *******************************************************************************
 *
 * @param[in,out] handle
 *          On entry, the handle to destroy.
 *          On exit, the handle cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zunmqr_New
 * @sa dplasma_zunmqr
 *
 ******************************************************************************/
void
dplasma_zunmqr_Destruct( parsec_taskpool_t *tp )
{
    parsec_zunmqr_LC_taskpool_t *parsec_zunmqr = (parsec_zunmqr_LC_taskpool_t *)tp;

    parsec_matrix_del2arena( parsec_zunmqr->arenas[PARSEC_zunmqr_LC_DEFAULT_ARENA   ] );
    parsec_matrix_del2arena( parsec_zunmqr->arenas[PARSEC_zunmqr_LC_LOWER_TILE_ARENA] );
    parsec_matrix_del2arena( parsec_zunmqr->arenas[PARSEC_zunmqr_LC_LITTLE_T_ARENA  ] );

    parsec_private_memory_fini( parsec_zunmqr->_g_pool_0 );
    free( parsec_zunmqr->_g_pool_0 );

    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zunmqr_New - Overwrites the general M-by-N matrix C with
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
 *  as returned by dplasma_zgeqrf(). Q is of order M if side = PlasmaLeft
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
 * @param[in] A
 *          Descriptor of the matrix A of size M-by-K if side == PlasmaLeft, or
 *          N-by-K if side == PlasmaRight factorized with the
 *          dplasma_zgeqrf_New() routine.
 *          On entry, the i-th column must contain the vector which
 *          defines the elementary reflector H(i), for i = 1,2,...,k, as
 *          returned by dplasma_zgeqrf_New() in the first k columns of its array
 *          argument A.
 *          If side == PlasmaLeft,  M >= K >= 0.
 *          If side == PlasmaRight, N >= K >= 0.
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile QR algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized during the call to dplasma_zgeqrf_New().
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
 * @sa dplasma_zunmqr_Destruct
 * @sa dplasma_zunmqr
 * @sa dplasma_cunmqr_New
 * @sa dplasma_dormqr_New
 * @sa dplasma_sormqr_New
 * @sa dplasma_zgeqrf_New
 *
 ******************************************************************************/
int
dplasma_zunmqr( parsec_context_t *parsec,
                PLASMA_enum side, PLASMA_enum trans,
                tiled_matrix_desc_t *A,
                tiled_matrix_desc_t *T,
                tiled_matrix_desc_t *C )
{
    parsec_taskpool_t *parsec_zunmqr = NULL;
    int Am;

    if (parsec == NULL) {
        dplasma_error("dplasma_zunmqr", "dplasma not initialized");
        return -1;
    }

    if ((side != PlasmaLeft) && (side != PlasmaRight)) {
        dplasma_error("dplasma_zunmqr", "illegal value of side");
        return -1;
    }
    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zunmqr", "illegal value of trans");
        return -2;
    }

    if ( side == PlasmaLeft ) {
        Am = C->m;
    } else {
        Am = C->n;
    }
    if ( A->m != Am ) {
        dplasma_error("dplasma_zunmqr", "illegal value of A->m");
        return -3;
    }
    if ( A->n > Am ) {
        dplasma_error("dplasma_zunmqr", "illegal value of A->n");
        return -5;
    }
    if ( (T->nt != A->nt) || (T->mt != A->mt) ) {
        dplasma_error("dplasma_zunmqr", "illegal size of T (T should have as many tiles as A)");
        return -20;
    }

    if (dplasma_imin(C->m, dplasma_imin(C->n, A->n)) == 0)
        return 0;

    parsec_zunmqr = dplasma_zunmqr_New(side, trans, A, T, C);

    if ( parsec_zunmqr != NULL ){
        parsec_enqueue(parsec, parsec_zunmqr);
        dplasma_wait_until_completion(parsec);
        dplasma_zunmqr_Destruct( parsec_zunmqr );
    }
    return 0;
}
