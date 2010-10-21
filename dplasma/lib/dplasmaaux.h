#ifndef DPLASMA_AUX_H_HAS_BEEN_INCLUDED
#define DPLASMA_AUX_H_HAS_BEEN_INCLUDED

/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include <dague.h>
#include "dplasma.h"

int dplasma_aux_get_priority( char* function, const tiled_matrix_desc_t* ddesc );

int dplasma_aux_create_rectangle_type( dague_remote_dep_datatype_t oldtype,
                                       unsigned int tile_nb,
                                       unsigned int tile_mb,
                                       int resized,
                                       dague_remote_dep_datatype_t* newtype );
int dplasma_aux_create_tile_type( dague_remote_dep_datatype_t oldtype,
                                  unsigned int tile_nb,
                                  dague_remote_dep_datatype_t* newtype );
int dplasma_aux_create_upper_type( dague_remote_dep_datatype_t oldtype,
                                   unsigned int tile_nb,
                                   dague_remote_dep_datatype_t* newtype );
int dplasma_aux_create_lower_type( dague_remote_dep_datatype_t oldtype,
                                   unsigned int tile_nb,
                                   dague_remote_dep_datatype_t* newtype );

#endif  /* DPLASMA_AUX_H_HAS_BEEN_INCLUDED */
