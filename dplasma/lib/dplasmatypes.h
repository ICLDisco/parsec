#ifndef DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED
#define DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED

/*
 * Copyright (c) 2010-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec.h"
#include "dplasma.h"
#include "parsec/arena.h"

#define dplasma_progress( object )              \
  parsec_context_start( object );               \
  parsec_context_wait( object );

static inline int
dplasma_add2arena_rectangle( parsec_arena_t *arena, size_t elem_size, size_t alignment,
                             parsec_datatype_t oldtype,
                             unsigned int tile_mb, unsigned int tile_nb, int resized )
{
    (void)elem_size;
    return parsec_matrix_add2arena( arena, oldtype,
                                   matrix_UpperLower, 1, tile_mb, tile_nb, tile_mb,
                                   alignment, resized );
}

static inline int
dplasma_add2arena_tile( parsec_arena_t *arena, size_t elem_size, size_t alignment,
                        parsec_datatype_t oldtype, unsigned int tile_mb )
{
    (void)elem_size;
    return parsec_matrix_add2arena( arena, oldtype,
                                   matrix_UpperLower, 1, tile_mb, tile_mb, tile_mb,
                                   alignment, -1 );
}

static inline int
dplasma_add2arena_upper( parsec_arena_t *arena, size_t elem_size, size_t alignment,
                         parsec_datatype_t oldtype, unsigned int tile_mb, int diag )
{
    (void)elem_size;
    return parsec_matrix_add2arena( arena, oldtype,
                                   matrix_Upper, diag, tile_mb, tile_mb, tile_mb,
                                   alignment, -1 );
}

static inline int
dplasma_add2arena_lower( parsec_arena_t *arena, size_t elem_size, size_t alignment,
                         parsec_datatype_t oldtype, unsigned int tile_mb, int diag )
{
    (void)elem_size;
    return parsec_matrix_add2arena( arena, oldtype,
                                   matrix_Lower, diag, tile_mb, tile_mb, tile_mb,
                                   alignment, -1 );
}

#endif  /* DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED */
