/*
 * Copyright (c) 2011-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague/dague_internal.h"
#include "data_dist/matrix/matrix.h"
#include "dague/utils/output.h"
#include "dague/arena.h"
#include "reduce_col.h"
#include "reduce_row.h"

dague_handle_t*
dague_reduce_col_New( const tiled_matrix_desc_t* src,
                      tiled_matrix_desc_t* dest,
                      dague_operator_t operator,
                      void* op_data )
{
    dague_reduce_col_handle_t* handle;
    dague_datatype_t oldtype, newtype;

    assert(src->mtype == dest->mtype);
    if( -1 == dague_translate_matrix_type(src->mtype, &oldtype) ) {
        dague_debug_verbose(3, dague_debug_output, "Unknown matrix type %d.", src->mtype );
        return NULL;
    }
    dague_type_create_contiguous(src->mb*src->nb, oldtype, &newtype);
    handle = dague_reduce_col_new( src, dest, operator, op_data, 0, 0, src->lnt, src->lmt );

#ifdef HAVE_MPI
    {
        int extent;
        MPI_Type_size(newtype, &extent);
        dague_arena_construct(handle->arenas[DAGUE_reduce_col_DEFAULT_ARENA],
                              extent,
                              DAGUE_ARENA_ALIGNMENT_SSE,
                              newtype);
    }
#else
    dague_arena_construct(handle->arenas[DAGUE_reduce_col_DEFAULT_ARENA],
                          src->mb*src->nb,
                          DAGUE_ARENA_ALIGNMENT_SSE,
                          DAGUE_DATATYPE_NULL );
#endif

    return (dague_handle_t*)handle;
}

void dague_reduce_col_Destruct( dague_handle_t *o )
{
    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}

dague_handle_t*
dague_reduce_row_New( const tiled_matrix_desc_t* src,
                      tiled_matrix_desc_t* dest,
                      dague_operator_t operator,
                      void* op_data )
{
    dague_reduce_row_handle_t* handle;
    dague_datatype_t oldtype, newtype;

    assert(src->mtype == dest->mtype);
    if( -1 == dague_translate_matrix_type(src->mtype, &oldtype) ) {
        dague_debug_verbose(3, dague_debug_output, "Unknown matrix type %d.", src->mtype );
        return NULL;
    }
    dague_type_create_contiguous(src->mb*src->nb, oldtype, &newtype);
    handle = dague_reduce_row_new( src, dest, operator, op_data, 0, 0, src->lnt, src->lmt );

#ifdef HAVE_MPI
    {
        int extent;
        MPI_Type_size(newtype, &extent);
        dague_arena_construct(handle->arenas[DAGUE_reduce_row_DEFAULT_ARENA],
                          extent,
                          DAGUE_ARENA_ALIGNMENT_SSE,
                          newtype);
    }
#else
    dague_arena_construct(handle->arenas[DAGUE_reduce_row_DEFAULT_ARENA],
                          src->mb*src->nb,
                          DAGUE_ARENA_ALIGNMENT_SSE,
                          DAGUE_DATATYPE_NULL);
#endif
    return (dague_handle_t*)handle;
}

void dague_reduce_row_Destruct( dague_handle_t *o )
{
    DAGUE_INTERNAL_HANDLE_DESTRUCT(o);
}

