/*
 * Copyright (c) 2011-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/utils/output.h"
#include "parsec/arena.h"
#include "reduce_col.h"
#include "reduce_row.h"

static void
__parsec_reduce_col_destructor(parsec_reduce_col_taskpool_t* tp)
{
    parsec_type_free(&tp->arenas_datatypes[PARSEC_reduce_col_DEFAULT_ADT_IDX].opaque_dtt);
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_reduce_col_taskpool_t, parsec_taskpool_t,
                          NULL, __parsec_reduce_col_destructor);

parsec_taskpool_t*
parsec_reduce_col_New( const parsec_tiled_matrix_t* src,
                      parsec_tiled_matrix_t* dest,
                      parsec_operator_t operation,
                      void* op_data )
{
    parsec_reduce_col_taskpool_t* tp;
    parsec_datatype_t oldtype, newtype;
    ptrdiff_t lb, extent;

    tp = parsec_reduce_col_new( src, dest, operation, op_data, 0, 0, src->lnt, src->lmt );
    assert(src->mtype == dest->mtype);
    if( PARSEC_SUCCESS != parsec_translate_matrix_type(src->mtype, &oldtype) ) {
        parsec_debug_verbose(3, parsec_debug_output, "Unknown matrix type %d.", src->mtype );
        return NULL;
    }
    parsec_type_create_contiguous(src->mb*src->nb, oldtype, &newtype);
    parsec_type_extent(newtype, &lb, &extent);
    parsec_arena_datatype_construct(&tp->arenas_datatypes[PARSEC_reduce_col_DEFAULT_ADT_IDX],
                                    extent,
                                    PARSEC_ARENA_ALIGNMENT_SSE,
                                    newtype);

    return (parsec_taskpool_t*)tp;
}



static void
__parsec_reduce_row_destructor(parsec_reduce_row_taskpool_t* tp)
{
    parsec_type_free(&tp->arenas_datatypes[PARSEC_reduce_row_DEFAULT_ADT_IDX].opaque_dtt);
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_reduce_row_taskpool_t, parsec_taskpool_t,
                          NULL, __parsec_reduce_row_destructor);

parsec_taskpool_t*
parsec_reduce_row_New( const parsec_tiled_matrix_t* src,
                      parsec_tiled_matrix_t* dest,
                      parsec_operator_t operation,
                      void* op_data )
{
    parsec_reduce_row_taskpool_t* tp;
    parsec_datatype_t oldtype, newtype;
    ptrdiff_t lb, extent;

    tp = parsec_reduce_row_new( src, dest, operation, op_data, 0, 0, src->lnt, src->lmt );
    assert(src->mtype == dest->mtype);
    if( PARSEC_SUCCESS != parsec_translate_matrix_type(src->mtype, &oldtype) ) {
        parsec_debug_verbose(3, parsec_debug_output, "Unknown matrix type %d.", src->mtype );
        return NULL;
    }
    parsec_type_create_contiguous(src->mb*src->nb, oldtype, &newtype);
    parsec_type_extent(newtype, &lb, &extent);
    parsec_arena_datatype_construct(&tp->arenas_datatypes[PARSEC_reduce_row_DEFAULT_ADT_IDX],
                                    extent,
                                    PARSEC_ARENA_ALIGNMENT_SSE,
                                    newtype);
    return (parsec_taskpool_t*)tp;
}

