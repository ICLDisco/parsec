/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "parsec/parsec_config.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "zger.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_internal
 *
 *  dplasma_zger_internal_New - Generates the taskpool that performs the gerc or
 *      geru operation
 *  dplasma_zger_internal_Destruct - Destroy the taskpool generated through
 *      dplasma_zger_internal_New()
 *  dplasma_zger_internal - Performs the gerc or geru operation
 *
 *******************************************************************************
 *
 * @param[in] trans
 *          @arg PlasmaTrans: geru operation is performed
 *          @arg PlasmaConjTrans: gerc operation is performed
 *
 * @param[in] alpha
 *          Specifies the scalar alpha.
 *
 * @param[in] X
 *          Descriptor of the distributed vector X.
 *
 * @param[in] Y
 *          Descriptor of the distributed vector Y.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the data described by A is overwritten by the updated matrix.
 *
 ******************************************************************************/
static inline parsec_taskpool_t*
dplasma_zger_internal_New( int trans, parsec_complex64_t alpha,
                           const parsec_tiled_matrix_dc_t *X,
                           const parsec_tiled_matrix_dc_t *Y,
                           parsec_tiled_matrix_dc_t *A)
{
    parsec_zger_taskpool_t* zger_tp;

    /* Check input arguments */
    if ((trans != PlasmaTrans) && (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zger", "illegal value of trans");
        return NULL /*-1*/;
    }
    zger_tp = parsec_zger_new(trans, alpha,
                              X,
                              Y,
                              A);

    dplasma_add2arena_tile( zger_tp->arenas[PARSEC_zger_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(parsec_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, A->mb);

    dplasma_add2arena_rectangle( zger_tp->arenas[PARSEC_zger_VECTOR_ARENA],
                                 X->mb*sizeof(parsec_complex64_t),
                                 PARSEC_ARENA_ALIGNMENT_SSE,
                                 parsec_datatype_double_complex_t, X->mb, 1, -1);

    return (parsec_taskpool_t*)zger_tp;
}

static inline void
dplasma_zger_internal_Destruct( parsec_taskpool_t *tp )
{
    parsec_matrix_del2arena( ((parsec_zger_taskpool_t *)tp)->arenas[PARSEC_zger_DEFAULT_ARENA] );
    parsec_matrix_del2arena( ((parsec_zger_taskpool_t *)tp)->arenas[PARSEC_zger_VECTOR_ARENA] );

    parsec_taskpool_free(tp);
}

static inline int
dplasma_zger_internal( parsec_context_t *parsec,
                       const int trans,
                       const parsec_complex64_t alpha,
                       const parsec_tiled_matrix_dc_t *X,
                       const parsec_tiled_matrix_dc_t *Y,
                             parsec_tiled_matrix_dc_t *A)
{
    parsec_taskpool_t *parsec_zger = NULL;

    /* Check input arguments */
    if ((trans != PlasmaTrans) && (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zger", "illegal value of trans");
        return -1;
    }

    parsec_zger = dplasma_zger_internal_New(trans, alpha, X, Y, A);

    if ( parsec_zger != NULL )
    {
        parsec_enqueue( parsec, parsec_zger);
        dplasma_wait_until_completion(parsec);
        dplasma_zger_internal_Destruct( parsec_zger );
        return 0;
    }
    else {
        return -101;
    }
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeru_New - Generates the taskpool that performs one of the following
 *  vector-matrix operations
 *
 *    \f[ A = \alpha [ X \times Y' ] + A \f],
 *
 *  where alpha is a scalar, x is an m element vector, y is an n element
 *  vector and A is an m by n matrix.
 *
 *  WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] alpha
 *          Specifies the scalar alpha.
 *
 * @param[in] X
 *          Descriptor of the distributed vector X.
 *
 * @param[in] Y
 *          Descriptor of the distributed vector Y.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the data described by A is overwritten by the updated matrix.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgeru_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeru
 * @sa dplasma_zgeru_Destruct
 * @sa dplasma_cgeru_New
 * @sa dplasma_dgeru_New
 * @sa dplasma_sgeru_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zgeru_New( const parsec_complex64_t alpha,
                   const parsec_tiled_matrix_dc_t *X,
                   const parsec_tiled_matrix_dc_t *Y,
                         parsec_tiled_matrix_dc_t *A)
{
    return dplasma_zger_internal_New( PlasmaTrans, alpha, X, Y, A );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeru_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zgeru_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpoll to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeru_New
 * @sa dplasma_zgeru
 *
 ******************************************************************************/
void
dplasma_zgeru_Destruct( parsec_taskpool_t *tp )
{
    dplasma_zger_internal_Destruct(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgeru - Performs one of the following vector-matrix operations
 *
 *    \f[ A = \alpha [ X \times Y' ] + A \f],
 *
 *  where alpha is a scalar, x is an m element vector, y is an n element
 *  vector and A is an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] alpha
 *          Specifies the scalar alpha.
 *
 * @param[in] X
 *          Descriptor of the distributed vector X.
 *
 * @param[in] Y
 *          Descriptor of the distributed vector Y.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the data described by A is overwritten by the updated matrix.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeru_New
 * @sa dplasma_zgeru_Destruct
 * @sa dplasma_cgeru
 * @sa dplasma_dgeru
 * @sa dplasma_sgeru
 *
 ******************************************************************************/
int
dplasma_zgeru( parsec_context_t *parsec,
               const parsec_complex64_t alpha,
               const parsec_tiled_matrix_dc_t *X,
               const parsec_tiled_matrix_dc_t *Y,
                     parsec_tiled_matrix_dc_t *A)
{
    return dplasma_zger_internal( parsec, PlasmaTrans, alpha, X, Y, A );
}

#if defined(PRECISION_z) || defined(PRECISION_c)

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgerc_New - Generates the taskpool that performs one of the following
 *  vector-matrix operations
 *
 *    \f[ A = \alpha [ X \times conj( Y' ) ] + A \f],
 *
 *  where alpha is a scalar, x is an m element vector, y is an n element
 *  vector and A is an m by n matrix.
 *
 *  WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] alpha
 *          Specifies the scalar alpha.
 *
 * @param[in] X
 *          Descriptor of the distributed vector X.
 *
 * @param[in] Y
 *          Descriptor of the distributed vector Y.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the data described by A is overwritten by the updated matrix.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_enqueue(). It, then, needs to be
 *          destroy with dplasma_zgerc_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgerc
 * @sa dplasma_zgerc_Destruct
 * @sa dplasma_cgerc_New
 * @sa dplasma_dgerc_New
 * @sa dplasma_sgerc_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zgerc_New( parsec_complex64_t alpha,
                   const parsec_tiled_matrix_dc_t *X,
                   const parsec_tiled_matrix_dc_t *Y,
                         parsec_tiled_matrix_dc_t *A)
{
    return dplasma_zger_internal_New( PlasmaConjTrans, alpha, X, Y, A );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgerc_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zgerc_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgerc_New
 * @sa dplasma_zgerc
 *
 ******************************************************************************/
void
dplasma_zgerc_Destruct( parsec_taskpool_t *tp )
{
    dplasma_zger_internal_Destruct(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgerc - Performs one of the following vector-matrix operations
 *
 *    \f[ A = \alpha [ X \times conj( Y' ) ] + A \f],
 *
 *  where alpha is a scalar, x is an m element vector, y is an n element
 *  vector and A is an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] alpha
 *          Specifies the scalar alpha.
 *
 * @param[in] X
 *          Descriptor of the distributed vector X.
 *
 * @param[in] Y
 *          Descriptor of the distributed vector Y.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A.
 *          On exit, the data described by A is overwritten by the updated matrix.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgerc_New
 * @sa dplasma_zgerc_Destruct
 * @sa dplasma_cgerc
 * @sa dplasma_dgerc
 * @sa dplasma_sgerc
 *
 ******************************************************************************/
int
dplasma_zgerc( parsec_context_t *parsec,
               parsec_complex64_t alpha,
               const parsec_tiled_matrix_dc_t *X,
               const parsec_tiled_matrix_dc_t *Y,
                     parsec_tiled_matrix_dc_t *A)
{
    return dplasma_zger_internal( parsec, PlasmaConjTrans, alpha, X, Y, A );
}

#endif /* defined(PRECISION_z) || defined(PRECISION_c) */
