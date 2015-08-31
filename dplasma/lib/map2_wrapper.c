/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 */

#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map2.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma
 *
 * dplasma_map2_New - Generates an object that performs a map operation with
 * two similar matrices, applying the operator on each couple of tiles:
 *
 *    operator( A, B )
 *
 * where A is constant and B receive the computation
 * If this operator is used with both matrices stored as Lapack Layout (or
 * Column Major), they must have the same distribution. Otherwise the matrix A,
 * can be only stored in tile layout.
 *
 * See dplasma_zlacpy_New(), dplasma_zgeadd_New() or dplasma_ztradd_New() as
 * example of function using the map2 operator.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies on which part of matrices A and B, the operator must be
 *          applied
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is referenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is non-transposed or transposed when
 *          given to the map function
 *          = PlasmaNoTrans:   map( A(m,n), B(m,n) )
 *          = PlasmaTrans:     map( A(n,m), B(m,n) )
 *          = PlasmaConjTrans: map( A(n,m), B(m,n) ), and conj is forwarded to
 *          the operator if required.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A. Any tiled matrix descriptor
 *          can be used. However, if the data is stored in column major, the
 *          tile distribution must match the one of the matrix B.
 *          Matrix A is untouched during computation.
 *
 * @param[in,out] B
 *          Descriptor of the distributed matrix B. Any tiled matrix descriptor
 *          can be used, with no specific storage.
 *
 * @param[in] operator
 *          Binary operator describing the operation to perform on each couple
 *          of tiles. The parameters of the descriptor are:
 *          operator( execution_context, tileA, tileB, op_args, uplo, m, n),
 *          where execution_context is the execution context that runs the
 *          operator, tileA, the pointer to the tile of matrix A, tileB the
 *          pointer to the tile of matrix B, op_args the parameters given to
 *          each operator call, uplo the part of the tiles on which the operator
 *          is working, and (m,n) the tile indices.
 *
 * @param[in] op_args
 *          Arguments given to each call to the binary operator.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_map2_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_map2
 * @sa dplasma_map2_Destruct
 *
 ******************************************************************************/
dague_handle_t *
dplasma_map2_New( PLASMA_enum uplo,
                  PLASMA_enum trans,
                  const tiled_matrix_desc_t *A,
                  tiled_matrix_desc_t *B,
                  tiled_matrix_binary_op_t operator,
                  void *op_args)
{
    dague_map2_handle_t *dague_map2 = NULL;

    if ((uplo != PlasmaUpperLower) &&
        (uplo != PlasmaUpper)      &&
        (uplo != PlasmaLower))
    {
        dplasma_error("dplasma_map2", "illegal value of uplo");
        return NULL;
    }

    if ((trans != PlasmaNoTrans) &&
        (trans != PlasmaTrans)   &&
        (trans != PlasmaConjTrans))
    {
        dplasma_error("dplasma_map2", "illegal value of trans");
        return NULL;
    }

    if ( ((trans == PlasmaNoTrans) &&
          ((A->mt != B->mt) || (A->nt != B->nt))) ||
         ((trans != PlasmaNoTrans) &&
          ((A->nt != B->mt) || (A->mt != B->nt))) )
    {
        dplasma_error("dplasma_map2", "dimension of A and B don't match");
        return NULL;
    }

    dague_map2 = dague_map2_new( uplo, trans,
                                 (dague_ddesc_t*)A,
                                 (dague_ddesc_t*)B,
                                 operator, op_args );

    switch( A->mtype ) {
    case matrix_ComplexDouble :
        dplasma_add2arena_tile( dague_map2->arenas[DAGUE_map2_DEFAULT_ARENA],
                                A->mb*A->nb*sizeof(dague_complex64_t),
                                DAGUE_ARENA_ALIGNMENT_SSE,
                                dague_datatype_double_complex_t, A->mb);
        break;
    case matrix_ComplexFloat  :
        dplasma_add2arena_tile( dague_map2->arenas[DAGUE_map2_DEFAULT_ARENA],
                                A->mb*A->nb*sizeof(dague_complex32_t),
                                DAGUE_ARENA_ALIGNMENT_SSE,
                                dague_datatype_complex_t, A->mb);
        break;
    case matrix_RealDouble    :
        dplasma_add2arena_tile( dague_map2->arenas[DAGUE_map2_DEFAULT_ARENA],
                                A->mb*A->nb*sizeof(double),
                                DAGUE_ARENA_ALIGNMENT_SSE,
                                dague_datatype_double_t, A->mb);
        break;
    case matrix_RealFloat     :
        dplasma_add2arena_tile( dague_map2->arenas[DAGUE_map2_DEFAULT_ARENA],
                                A->mb*A->nb*sizeof(float),
                                DAGUE_ARENA_ALIGNMENT_SSE,
                                dague_datatype_float_t, A->mb);
        break;
    case matrix_Integer       :
    default:
        dplasma_add2arena_tile( dague_map2->arenas[DAGUE_map2_DEFAULT_ARENA],
                                A->mb*A->nb*sizeof(int),
                                DAGUE_ARENA_ALIGNMENT_SSE,
                                dague_datatype_int_t, A->mb);
    }
    return (dague_handle_t*)dague_map2;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma
 *
 *  dplasma_map2_Destruct - Free the data structure associated to an object
 *  created with dplasma_map2_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_map2_New
 * @sa dplasma_map2
 *
 ******************************************************************************/
void
dplasma_map2_Destruct( dague_handle_t *o )
{
    dague_map2_handle_t *omap2 = (dague_map2_handle_t *)o;

    if ( omap2->op_args ) {
        free( omap2->op_args );
    }

    dague_matrix_del2arena( omap2->arenas[DAGUE_map2_DEFAULT_ARENA] );

    DAGUE_INTERNAL_HANDLE_DESTRUCT(omap2);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma
 *
 * dplasma_map2_New - Performs a map operation with two similar matrices,
 * applying the operator on each couple of tiles:
 *
 *    operator( A, B )
 *
 * where A is constant and B receive the computation
 * If this operator is used with both matrices stored as Lapack Layout (or
 * Column Major), they must have the same distribution. Otherwise the matrix A,
 * can be only stored in tile layout.
 *
 * See dplasma_zlacpy() or dplasma_zgeadd() as example of function using
 * the map2 operator.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies on which part of matrices A and B, the operator must be
 *          applied
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is referenced.
 *          = PlasmaLower:      Only lower part is referenced.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A. Any tiled matrix descriptor
 *          can be used. However, if the data is stored in column major, the
 *          tile distribution must match the one of the matrix B.
 *
 * @param[in,out] B
 *          Descriptor of the distributed matrix B. Any tiled matrix descriptor
 *          can be used, with no specific storage.
 *
 * @param[in] operator
 *          Binary operator describing the operation to perform on each couple
 *          of tiles. The parameters of the descriptor are:
 *          operator( execution_context, tileA, tileB, op_args, uplo, m, n),
 *          where execution_context is the execution context that runs the
 *          operator, tileA, the pointer to the tile of matrix A, tileB the
 *          pointer to the tile of matrix B, op_args the parameters given to
 *          each operator call, uplo the part of the tiles on which the operator
 *          is working, and (m,n) the tile indices.
 *
 * @param[in] op_args
 *          Arguments given to each call to the binary operator.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_map2_New
 * @sa dplasma_map2_Destruct
 *
 ******************************************************************************/
int
dplasma_map2( dague_context_t *dague,
              PLASMA_enum uplo,
              PLASMA_enum trans,
              const tiled_matrix_desc_t *A,
              tiled_matrix_desc_t *B,
              tiled_matrix_binary_op_t operator,
              void *op_args)
{
    dague_handle_t *dague_map2 = NULL;

    if ((uplo != PlasmaUpperLower) &&
        (uplo != PlasmaUpper)      &&
        (uplo != PlasmaLower))
    {
        dplasma_error("dplasma_map2", "illegal value of uplo");
        return -2;
    }

    dague_map2 = dplasma_map2_New( uplo, trans, A, B, operator, op_args );

    if ( dague_map2 != NULL )
    {
        dague_enqueue( dague, dague_map2 );
        dplasma_progress( dague );
        dplasma_map2_Destruct( dague_map2 );
    }

    return 0;
}
