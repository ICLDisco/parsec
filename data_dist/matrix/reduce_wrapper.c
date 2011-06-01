/*
 * Copyright (c) 2011      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague.h"
#include "data_dist/matrix/matrix.h"
#include "generated/reduce_col.h"
#include "generated/reduce_row.h"

struct dague_object_t*
dague_reduce_col_New( tiled_matrix_desc_t* A,
                      tiled_matrix_desc_t* res,
                      dague_operator_t operator,
                      void* op_data )
{
    struct dague_object_t* dague;

    dague = (struct dague_object_t*)dague_reduce_col_new( A->lnt, A, res, operator, op_data, 0, 0, A->lnt, A->lmt );
    return dague;
}

void dague_reduce_col_Destruct( struct dague_object_t *o )
{
    dague_reduce_col_destroy( (dague_reduce_col_object_t*)o );
}

struct dague_object_t*
dague_reduce_row_New( tiled_matrix_desc_t* A,
                      tiled_matrix_desc_t* res,
                      dague_operator_t operator,
                      void* op_data )
{
    struct dague_object_t* dague;

    dague = (struct dague_object_t*)dague_reduce_row_new( A->lnt, A, res, operator, op_data );
    return dague;
}

void dague_reduce_row_Destruct( struct dague_object_t *o )
{
    dague_reduce_row_destroy( (dague_reduce_row_object_t*)o );
}

