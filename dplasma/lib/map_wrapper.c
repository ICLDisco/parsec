/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 */
#include "dague_internal.h"
#include "dplasma.h"
#include "dplasma/lib/dplasmatypes.h"

#include "map.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma
 *
 * dplasma_map_New - Generates an object that performs a map operation with
 * two similar matrices, applying the operator on each tile of A:
 *
 *    operator( A )
 *
 * See dplasma_zlaset_New() or dplasma_zlascal_New() as example of function using
 * the map operator.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies on which part of matrix A, the operator must be
 *          applied
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is refrenced.
 *          = PlasmaLower:      Only lower part is referenced.
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
 *          \retval The dague object describing the operation that can be
 *          enqueued in the runtime with dague_enqueue(). It, then, needs to be
 *          destroy with dplasma_map_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_map
 * @sa dplasma_map_Destruct
 *
 ******************************************************************************/
dague_object_t *
dplasma_map_New( PLASMA_enum uplo,
                 tiled_matrix_desc_t *A,
                 tiled_matrix_unary_op_t operator,
                 void *op_args )
{
    dague_map_object_t *dague_map = NULL;

    if ((uplo != PlasmaUpperLower) &&
        (uplo != PlasmaUpper)      &&
        (uplo != PlasmaLower))
    {
        dplasma_error("dplasma_map", "illegal value of uplo");
        return NULL;
    }

   dague_map = dague_map_new( uplo,
                                 (dague_ddesc_t*)A,
                                 operator, op_args);

    switch( A->mtype ) {
    case matrix_ComplexDouble :
        dplasma_add2arena_tile( dague_map->arenas[DAGUE_map_DEFAULT_ARENA],
                                A->mb*A->nb*sizeof(dague_complex64_t),
                                DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_DOUBLE_COMPLEX, A->mb);
        break;
    case matrix_ComplexFloat  :
        dplasma_add2arena_tile( dague_map->arenas[DAGUE_map_DEFAULT_ARENA],
                                A->mb*A->nb*sizeof(dague_complex32_t),
                                DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_COMPLEX, A->mb);
        break;
    case matrix_RealDouble    :
        dplasma_add2arena_tile( dague_map->arenas[DAGUE_map_DEFAULT_ARENA],
                                A->mb*A->nb*sizeof(double),
                                DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_DOUBLE, A->mb);
        break;
    case matrix_RealFloat     :
        dplasma_add2arena_tile( dague_map->arenas[DAGUE_map_DEFAULT_ARENA],
                                A->mb*A->nb*sizeof(float),
                                DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_FLOAT, A->mb);
        break;
    case matrix_Integer       :
    default:
        dplasma_add2arena_tile( dague_map->arenas[DAGUE_map_DEFAULT_ARENA],
                                A->mb*A->nb*sizeof(int),
                                DAGUE_ARENA_ALIGNMENT_SSE,
                                MPI_INT, A->mb);
    }
    return (dague_object_t*)dague_map;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma
 *
 *  dplasma_map_Destruct - Free the data structure associated to an object
 *  created with dplasma_map_New().
 *
 *******************************************************************************
 *
 * @param[in,out] o
 *          On entry, the object to destroy.
 *          On exit, the object cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_map_New
 * @sa dplasma_map
 *
 ******************************************************************************/
void
dplasma_map_Destruct( dague_object_t *o )
{
    dague_map_object_t *omap = (dague_map_object_t *)o;

    if ( omap->op_args ) {
        free( omap->op_args );
    }

    dplasma_datatype_undefine_type( &(omap->arenas[DAGUE_map_DEFAULT_ARENA]->opaque_dtt) );

    DAGUE_INTERNAL_OBJECT_DESTRUCT(omap);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma
 *
 * dplasma_map_New - Performs a map operation with two similar matrices,
 * applying the operator on each tile of A:
 *
 *    operator( A )
 *
 * See dplasma_zlaset() or dplasma_zlascal() as example of function using
 * the map operator.
 *
 * WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in,out] dague
 *          The dague context of the application that will run the operation.
 *
 * @param[in] uplo
 *          Specifies on which part of matrix A, the operator must be
 *          applied
 *          = PlasmaUpperLower: All matrix is referenced.
 *          = PlasmaUpper:      Only upper part is refrenced.
 *          = PlasmaLower:      Only lower part is referenced.
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
 *******************************************************************************
 *
 * @sa dplasma_map_New
 * @sa dplasma_map_Destruct
 *
 ******************************************************************************/
int
dplasma_map( dague_context_t *dague,
             PLASMA_enum uplo,
             tiled_matrix_desc_t *A,
             tiled_matrix_unary_op_t operator,
             void *op_args )
{
    dague_object_t *dague_map = NULL;

    if ((uplo != PlasmaUpperLower) &&
        (uplo != PlasmaUpper)      &&
        (uplo != PlasmaLower))
    {
        dplasma_error("dplasma_map", "illegal value of uplo");
        return -2;
    }

    dague_map = dplasma_map_New( uplo, A, operator, op_args );

    if ( dague_map != NULL )
    {
        dague_enqueue( dague, dague_map );
        dplasma_progress( dague );
        dplasma_map_Destruct( dague_map );
    }

    return 0;
}
