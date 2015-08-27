#ifndef DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED
#define DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED

/*
 * Copyright (c) 2010-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "dague.h"
#include "dplasma.h"
#include "dague/arena.h"

#define dplasma_comm MPI_COMM_WORLD

#if defined(HAVE_MPI)

#define dplasma_progress( object )                  \
    do {                                            \
        /*MPI_Barrier(dplasma_comm);*/              \
        dague_context_wait( object );               \
    } while (0)

#else

#define dplasma_progress( object )              \
  dague_context_wait( object );

#endif

static inline int
dplasma_add2arena_rectangle( dague_arena_t *arena, size_t elem_size, size_t alignment,
                             dague_datatype_t oldtype,
                             unsigned int tile_mb, unsigned int tile_nb, int resized )
{
    (void)elem_size;
    return dague_matrix_add2arena( arena, oldtype,
                                   matrix_UpperLower, 1, tile_mb, tile_nb, tile_mb,
                                   alignment, resized );
}

static inline int
dplasma_add2arena_tile( dague_arena_t *arena, size_t elem_size, size_t alignment,
                        dague_datatype_t oldtype, unsigned int tile_mb )
{
    (void)elem_size;
    return dague_matrix_add2arena( arena, oldtype,
                                   matrix_UpperLower, 1, tile_mb, tile_mb, tile_mb,
                                   alignment, -1 );
}

static inline int
dplasma_add2arena_upper( dague_arena_t *arena, size_t elem_size, size_t alignment,
                         dague_datatype_t oldtype, unsigned int tile_mb, int diag )
{
    (void)elem_size;
    return dague_matrix_add2arena( arena, oldtype,
                                   matrix_Upper, diag, tile_mb, tile_mb, tile_mb,
                                   alignment, -1 );
}

static inline int
dplasma_add2arena_lower( dague_arena_t *arena, size_t elem_size, size_t alignment,
                         dague_datatype_t oldtype, unsigned int tile_mb, int diag )
{
    (void)elem_size;
    return dague_matrix_add2arena( arena, oldtype,
                                   matrix_Lower, diag, tile_mb, tile_mb, tile_mb,
                                   alignment, -1 );
}

#endif  /* DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED */
