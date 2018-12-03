/*
 * Copyright (c) 2018      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/data_dist/matrix/matrix.h"
#include "apply.h"

/**
 *******************************************************************************
 * apply_New - Generates an taskpool that applies the operator on each tile of A:
 *
 *    operator( A )
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies on which part of matrix A, the operator must be
 *          applied
 *          = matrix_UpperLower: All matrix is referenced.
 *          = matrix_Upper:      Only upper part is refrenced.
 *          = matrix_Lower:      Only lower part is referenced.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A on which operator is applied.
 *
 * @param[in] operator
 *          Unary operator describing the operation to perform on each couple of
 *          tiles. The parameters of the descriptor are: operator(
 *          execution_context, tileA, op_args, uplo, m, n), where
 *          execution_context is the execution context that runs the operator,
 *          tileA, the pointer to the tile of matrix A, op_args the parameters
 *          given to each operator call, uplo the part of the tiles on which the
 *          operator is working, and (m,n) the tile indices.
 *
 * @param[in] op_args
 *          Arguments given to each call to the unary operator.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with apply_Destruct();
 *
 */

parsec_taskpool_t *
parsec_apply_New( int uplo,
                 parsec_tiled_matrix_dc_t *A,
                 tiled_matrix_unary_op_t operator,
                 void *op_args )
{
    parsec_apply_taskpool_t *parsec_app = NULL;

    if ((uplo != matrix_UpperLower) &&
        (uplo != matrix_Upper)      &&
        (uplo != matrix_Lower))
    {
        return NULL;
    }

    parsec_app =   parsec_apply_new( uplo,
                                     A,
                                     operator, op_args);

    switch( A->mtype ) {
    case matrix_ComplexDouble    :
        parsec_matrix_add2arena( parsec_app->arenas[PARSEC_apply_DEFAULT_ARENA], 
                                 parsec_datatype_double_complex_t,
                                 matrix_UpperLower, 1, A->mb, A->mb, A->mb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
        break;
    case matrix_ComplexFloat     :
        parsec_matrix_add2arena( parsec_app->arenas[PARSEC_apply_DEFAULT_ARENA], 
                                 parsec_datatype_complex_t,
                                 matrix_UpperLower, 1, A->mb, A->mb, A->mb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
        break;
    case matrix_RealDouble       :
        parsec_matrix_add2arena( parsec_app->arenas[PARSEC_apply_DEFAULT_ARENA], parsec_datatype_double_t,
                                 matrix_UpperLower, 1, A->mb, A->mb, A->mb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
        break;
    case matrix_RealFloat        :
        parsec_matrix_add2arena( parsec_app->arenas[PARSEC_apply_DEFAULT_ARENA], parsec_datatype_float_t,
                                 matrix_UpperLower, 1, A->mb, A->mb, A->mb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
        break;
    case matrix_Integer          :
    default:
        parsec_matrix_add2arena( parsec_app->arenas[PARSEC_apply_DEFAULT_ARENA], parsec_datatype_int_t,
                                 matrix_UpperLower, 1, A->mb, A->mb, A->mb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
    }
    return (parsec_taskpool_t*)parsec_app;
}

/**
 *******************************************************************************
 *
 *  apply_Destruct - Free the data structure associated to an taskpool
 *  created with apply_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 ******************************************************************************/
void
parsec_apply_Destruct( parsec_taskpool_t *tp )
{
    parsec_apply_taskpool_t *omap = (parsec_apply_taskpool_t *)tp;

    if ( omap->_g_op_args ) {
        free( omap->_g_op_args );
    }

    parsec_matrix_del2arena( omap->arenas[PARSEC_apply_DEFAULT_ARENA] );

    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 * apply_New - Performs apply operation on each tile of A:
 *
 *    operator( A )
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies on which part of matrix A, the operator must be
 *          applied
 *          = matrix_UpperLower: All matrix is referenced.
 *          = matrix_Upper:      Only upper part is refrenced.
 *          = matrix_Lower:      Only lower part is referenced.
 *
 * @param[in,out] A
 *          Descriptor of the distributed matrix A on which operator is applied.
 *
 * @param[in] operator
 *          Unary operator describing the operation to perform on each couple of
 *          tiles. The parameters of the descriptor are: operator(
 *          execution_context, tileA, op_args, uplo, m, n), where
 *          execution_context is the execution context that runs the operator,
 *          tileA, the pointer to the tile of matrix A, op_args the parameters
 *          given to each operator call, uplo the part of the tiles on which the
 *          operator is working, and (m,n) the tile indices.
 *
 * @param[in] op_args
 *          Arguments given to each call to the unary operator.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 ******************************************************************************/
int
parsec_apply( parsec_context_t *parsec,
             int uplo,
             parsec_tiled_matrix_dc_t *A,
             tiled_matrix_unary_op_t operator,
             void *op_args )
{
    parsec_taskpool_t *parsec_app = NULL;

    if ((uplo != matrix_UpperLower) &&
        (uplo != matrix_Upper)      &&
        (uplo != matrix_Lower))
    {
        return -2;
    }

    parsec_app = parsec_apply_New( uplo, A, operator, op_args );

    if ( parsec_app != NULL )
    {
        parsec_context_add_taskpool( parsec, parsec_app );
        parsec_context_start( parsec );
        parsec_context_wait( parsec );
        parsec_apply_Destruct( parsec_app );
    }

    return 0;
}
