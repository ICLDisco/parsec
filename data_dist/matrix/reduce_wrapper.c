/*
 * Copyright (c) 2011-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "data_dist/matrix/matrix.h"
#include "parsec/utils/output.h"
#include "parsec/arena.h"
#include "reduce_col.h"
#include "reduce_row.h"

parsec_taskpool_t*
parsec_reduce_col_New( const tiled_matrix_desc_t* src,
                      tiled_matrix_desc_t* dest,
                      parsec_operator_t operator,
                      void* op_data )
{
    parsec_reduce_col_taskpool_t* tp;
    parsec_datatype_t oldtype, newtype;
    ptrdiff_t lb, extent;

    tp = parsec_reduce_col_new( src, dest, operator, op_data, 0, 0, src->lnt, src->lmt );
    assert(src->mtype == dest->mtype);
    if( -1 == parsec_translate_matrix_type(src->mtype, &oldtype) ) {
        parsec_debug_verbose(3, parsec_debug_output, "Unknown matrix type %d.", src->mtype );
        return NULL;
    }
    parsec_type_create_contiguous(src->mb*src->nb, oldtype, &newtype);
    parsec_type_extent(newtype, &lb, &extent);
    parsec_arena_construct(tp->arenas[PARSEC_reduce_col_DEFAULT_ARENA],
                           extent,
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           newtype);

    return (parsec_taskpool_t*)tp;
}

void parsec_reduce_col_Destruct( parsec_taskpool_t *o )
{
    PARSEC_INTERNAL_TASKPOOL_DESTRUCT(o);
}

parsec_taskpool_t*
parsec_reduce_row_New( const tiled_matrix_desc_t* src,
                      tiled_matrix_desc_t* dest,
                      parsec_operator_t operator,
                      void* op_data )
{
    parsec_reduce_row_taskpool_t* tp;
    parsec_datatype_t oldtype, newtype;
    ptrdiff_t lb, extent;

    tp = parsec_reduce_row_new( src, dest, operator, op_data, 0, 0, src->lnt, src->lmt );
    assert(src->mtype == dest->mtype);
    if( -1 == parsec_translate_matrix_type(src->mtype, &oldtype) ) {
        parsec_debug_verbose(3, parsec_debug_output, "Unknown matrix type %d.", src->mtype );
        return NULL;
    }
    parsec_type_create_contiguous(src->mb*src->nb, oldtype, &newtype);
    parsec_type_extent(newtype, &lb, &extent);
    parsec_arena_construct(tp->arenas[PARSEC_reduce_row_DEFAULT_ARENA],
                           extent,
                           PARSEC_ARENA_ALIGNMENT_SSE,
                           newtype);
    return (parsec_taskpool_t*)tp;
}

void parsec_reduce_row_Destruct( parsec_taskpool_t *o )
{
    PARSEC_INTERNAL_TASKPOOL_DESTRUCT(o);
}

