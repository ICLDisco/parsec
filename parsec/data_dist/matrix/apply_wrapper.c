/*
 * Copyright (c) 2018-2022 The University of Tennessee and The University
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
 *          = PARSEC_MATRIX_FULL:  All matrix is referenced.
 *          = PARSEC_MATRIX_UPPER: Only upper part is refrenced.
 *          = PARSEC_MATRIX_LOWER: Only lower part is referenced.
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
parsec_apply_New( parsec_matrix_uplo_t uplo,
                 parsec_tiled_matrix_t *A,
                 parsec_tiled_matrix_unary_op_t operation,
                 void *op_args )
{
    parsec_apply_taskpool_t *parsec_app = NULL;

    if ((uplo != PARSEC_MATRIX_FULL) &&
        (uplo != PARSEC_MATRIX_UPPER)      &&
        (uplo != PARSEC_MATRIX_LOWER))
    {
        return NULL;
    }

    parsec_app =   parsec_apply_new( uplo,
                                     A,
                                     operation, op_args);

    switch( A->mtype ) {
    case PARSEC_MATRIX_COMPLEX_DOUBLE    :
        parsec_add2arena( &parsec_app->arenas_datatypes[PARSEC_apply_DEFAULT_ADT_IDX],
                          parsec_datatype_double_complex_t,
                          PARSEC_MATRIX_FULL, 1, A->mb, A->mb, A->mb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
        break;
    case PARSEC_MATRIX_COMPLEX_FLOAT     :
        parsec_add2arena( &parsec_app->arenas_datatypes[PARSEC_apply_DEFAULT_ADT_IDX],
                          parsec_datatype_complex_t,
                          PARSEC_MATRIX_FULL, 1, A->mb, A->mb, A->mb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
        break;
    case PARSEC_MATRIX_DOUBLE       :
        parsec_add2arena( &parsec_app->arenas_datatypes[PARSEC_apply_DEFAULT_ADT_IDX], parsec_datatype_double_t,
                          PARSEC_MATRIX_FULL, 1, A->mb, A->mb, A->mb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
        break;
    case PARSEC_MATRIX_FLOAT        :
        parsec_add2arena( &parsec_app->arenas_datatypes[PARSEC_apply_DEFAULT_ADT_IDX], parsec_datatype_float_t,
                          PARSEC_MATRIX_FULL, 1, A->mb, A->mb, A->mb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
        break;
    case PARSEC_MATRIX_INTEGER          :
    default:
        parsec_add2arena( &parsec_app->arenas_datatypes[PARSEC_apply_DEFAULT_ADT_IDX], parsec_datatype_int_t,
                          PARSEC_MATRIX_FULL, 1, A->mb, A->mb, A->mb, PARSEC_ARENA_ALIGNMENT_SSE, -1);
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

    parsec_del2arena( &omap->arenas_datatypes[PARSEC_apply_DEFAULT_ADT_IDX] );

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
 *          = PARSEC_MATRIX_FULL:  All matrix is referenced.
 *          = PARSEC_MATRIX_UPPER: Only upper part is refrenced.
 *          = PARSEC_MATRIX_LOWER: Only lower part is referenced.
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
 *          \retval PARSEC_ERR_BAD_PARAM if parameters are incorrect.
 *          \retval PARSEC_ERROR if an apply taskpool could not be created.
 *          \retval PARSEC_SUCCESS on success.
 *
 ******************************************************************************/
int
parsec_apply( parsec_context_t *parsec,
             parsec_matrix_uplo_t uplo,
             parsec_tiled_matrix_t *A,
             parsec_tiled_matrix_unary_op_t operation,
             void *op_args )
{
    parsec_taskpool_t *parsec_app = NULL;

    if ((uplo != PARSEC_MATRIX_FULL) &&
        (uplo != PARSEC_MATRIX_UPPER)      &&
        (uplo != PARSEC_MATRIX_LOWER))
    {
        return PARSEC_ERR_BAD_PARAM;
    }

    parsec_app = parsec_apply_New( uplo, A, operation, op_args );

    if ( parsec_app != NULL )
    {
        parsec_context_add_taskpool( parsec, parsec_app );
        parsec_context_start( parsec );
        parsec_context_wait( parsec );
        parsec_apply_Destruct( parsec_app );
    }
    else {
        return PARSEC_ERROR;
    }

    return PARSEC_SUCCESS;
}
