#ifndef DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED
#define DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED

/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <dague.h>
#include "dplasma.h"
#include "remote_dep.h"
#include "arena.h"

#define dplasma_comm MPI_COMM_WORLD

#if defined(HAVE_MPI)
/**
 * A portable accessor across all MPI versions (1.1 and 2.0) for
 * accessing the extent of a datatype.
 */
int dplasma_get_extent( MPI_Datatype dt, MPI_Aint* extent );
int dplasma_datatype_define_contiguous( dague_datatype_t oldtype,
                                        unsigned int nb_elem,
                                        int resized,
                                        dague_datatype_t* newtype );
int dplasma_datatype_define_rectangle( dague_datatype_t oldtype,
                                       unsigned int tile_mb,
                                       unsigned int tile_nb,
                                       int resized,
                                       dague_datatype_t* newtype );
int dplasma_datatype_define_tile( dague_datatype_t oldtype,
                                  unsigned int tile_nb,
                                  dague_datatype_t* newtype );
int dplasma_datatype_define_upper( dague_datatype_t oldtype,
                                   unsigned int tile_nb, int diag,
                                   dague_datatype_t* newtype );
int dplasma_datatype_define_lower( dague_datatype_t oldtype,
                                   unsigned int tile_nb, int diag,
                                   dague_datatype_t* newtype );

int dplasma_datatype_undefine_type(dague_datatype_t* type);

#define dplasma_progress( object )              \
    do {                                        \
        /*MPI_Barrier(dplasma_comm);*/              \
        dague_progress( object );               \
    } while (0)

#else
# define MPI_DOUBLE_COMPLEX NULL
# define MPI_COMPLEX        NULL
# define MPI_DOUBLE         NULL
# define MPI_FLOAT          NULL
# define MPI_INTEGER        NULL
# define MPI_INT            NULL

# define dplasma_datatype_define_contiguous( oldtype, nb_elem, resized, newtype) (*(newtype) = NULL)
# define dplasma_datatype_define_rectangle( oldtype, tile_mb, tile_nb, resized, newtype) (*(newtype) = NULL)
# define dplasma_datatype_define_tile(      oldtype, tile_nb, newtype ) (*(newtype) = NULL)
# define dplasma_datatype_define_upper(     oldtype, tile_nb, diag, newtype) (*(newtype) = NULL)
# define dplasma_datatype_define_lower(     oldtype, tile_nb, diag, newtype) (*(newtype) = NULL)
# define dplasma_datatype_undefine_type( type ) ( *(type) = NULL )

#define dplasma_progress( object )              \
  dague_progress( object );

#endif

int dplasma_add2arena_contiguous( dague_arena_t *arena, size_t elem_size, size_t alignment,
                                  dague_datatype_t oldtype,
                                  unsigned int nb_elem, int resized );
int dplasma_add2arena_rectangle( dague_arena_t *arena, size_t elem_size, size_t alignment,
                                 dague_datatype_t oldtype,
                                 unsigned int tile_mb, unsigned int tile_nb, int resized );
int dplasma_add2arena_tile( dague_arena_t *arena, size_t elem_size, size_t alignment,
                            dague_datatype_t oldtype, unsigned int tile_mb );
int dplasma_add2arena_upper( dague_arena_t *arena, size_t elem_size, size_t alignment,
                             dague_datatype_t oldtype, unsigned int tile_mb, int diag );
int dplasma_add2arena_lower( dague_arena_t *arena, size_t elem_size, size_t alignment,
                             dague_datatype_t oldtype, unsigned int tile_mb, int diag );

#endif  /* DPLASMA_DATATYPE_H_HAS_BEEN_INCLUDED */
