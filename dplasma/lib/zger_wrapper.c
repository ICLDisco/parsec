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

#include "zger.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_internal
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zger_internal_New - Generates the object that performs the gerc or
 *      geru operation
 *  dplasma_zger_internal_Destruct - Destroy the object generated through
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
static inline dague_object_t*
dplasma_zger_internal_New( int trans, dague_complex64_t alpha,
                           const tiled_matrix_desc_t *X,
                           const tiled_matrix_desc_t *Y,
                           tiled_matrix_desc_t *A)
{
    dague_zger_object_t* zger_object;

    /* Check input arguments */
    if ((trans != PlasmaTrans) && (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zger", "illegal value of trans");
        return NULL /*-1*/;
    }
    zger_object = dague_zger_new(trans, alpha,
                                 (dague_ddesc_t*)X,
                                 (dague_ddesc_t*)Y,
                                 (dague_ddesc_t*)A);

    dplasma_add2arena_tile( zger_object->arenas[DAGUE_zger_DEFAULT_ARENA],
                            A->mb*A->nb*sizeof(dague_complex64_t),
                            DAGUE_ARENA_ALIGNMENT_SSE,
                            MPI_DOUBLE_COMPLEX, A->mb);

    dplasma_add2arena_rectangle( zger_object->arenas[DAGUE_zger_VECTOR_ARENA],
                                 X->mb*sizeof(dague_complex64_t),
                                 DAGUE_ARENA_ALIGNMENT_SSE,
                                 MPI_DOUBLE_COMPLEX, X->mb, 1, -1);

    return (dague_object_t*)zger_object;
}

static inline void
dplasma_zger_internal_Destruct( dague_object_t *o )
{
    dplasma_datatype_undefine_type( &(((dague_zger_object_t *)o)->arenas[DAGUE_zger_DEFAULT_ARENA]->opaque_dtt) );
    dplasma_datatype_undefine_type( &(((dague_zger_object_t *)o)->arenas[DAGUE_zger_VECTOR_ARENA]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(o);
}

static inline int
dplasma_zger_internal( dague_context_t *dague,
                       const int trans,
                       const dague_complex64_t alpha,
                       const tiled_matrix_desc_t *X,
                       const tiled_matrix_desc_t *Y,
                             tiled_matrix_desc_t *A)
{
    dague_object_t *dague_zger = NULL;

    /* Check input arguments */
    if ((trans != PlasmaTrans) && (trans != PlasmaConjTrans)) {
        dplasma_error("dplasma_zger", "illegal value of trans");
        return -1;
    }

    dague_zger = dplasma_zger_internal_New(trans, alpha, X, Y, A);

    if ( dague_zger != NULL )
    {
        dague_enqueue( dague, dague_zger);
        dplasma_progress(dague);
        dplasma_zger_internal_Destruct( dague_zger );
        return 0;
    }
    else {
        return -101;
    }
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zgeru_New - Generates the object that performs one of the following
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
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
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
dague_object_t*
dplasma_zgeru_New( const dague_complex64_t alpha,
                   const tiled_matrix_desc_t *X,
                   const tiled_matrix_desc_t *Y,
                         tiled_matrix_desc_t *A)
{
    return dplasma_zger_internal_New( PlasmaTrans, alpha, X, Y, A );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zgeru_Destruct - Free the data structure associated to an object
 *  created with dplasma_zgeru_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgeru_New
 * @sa dplasma_zgeru
 *
 ******************************************************************************/
void
dplasma_zgeru_Destruct( dague_object_t *o )
{
    dplasma_zger_internal_Destruct( o );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
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
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
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
dplasma_zgeru( dague_context_t *dague,
               const dague_complex64_t alpha,
               const tiled_matrix_desc_t *X,
               const tiled_matrix_desc_t *Y,
                     tiled_matrix_desc_t *A)
{
    return dplasma_zger_internal( dague, PlasmaTrans, alpha, X, Y, A );
}

#if defined(PRECISION_z) || defined(PRECISION_c)

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zgerc_New - Generates the object that performs one of the following
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
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
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
dague_object_t*
dplasma_zgerc_New( dague_complex64_t alpha,
                   const tiled_matrix_desc_t *X,
                   const tiled_matrix_desc_t *Y,
                         tiled_matrix_desc_t *A)
{
    return dplasma_zger_internal_New( PlasmaConjTrans, alpha, X, Y, A );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
 *
 *  dplasma_zgerc_Destruct - Free the data structure associated to an object
 *  created with dplasma_zgerc_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgerc_New
 * @sa dplasma_zgerc
 *
 ******************************************************************************/
void
dplasma_zgerc_Destruct( dague_object_t *o )
{
    dplasma_zger_internal_Destruct( o );
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64_t
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
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
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
dplasma_zgerc( dague_context_t *dague,
               dague_complex64_t alpha,
               const tiled_matrix_desc_t *X,
               const tiled_matrix_desc_t *Y,
                     tiled_matrix_desc_t *A)
{
    return dplasma_zger_internal( dague, PlasmaConjTrans, alpha, X, Y, A );
}

#endif /* defined(PRECISION_z) || defined(PRECISION_c) */
