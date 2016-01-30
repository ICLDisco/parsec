/*
 * Copyright (c) 2014-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef VECTOR_H_HAS_BEEN_INCLUDED
#define VECTOR_H_HAS_BEEN_INCLUDED

#include "dague.h"
#include "dague/data_distribution.h"

dague_ddesc_t*
create_vector(int me, int world, int start_rank,
              int block_size, int total_size);
void release_vector(dague_ddesc_t *d);

#endif  /* VECTOR_H_HAS_BEEN_INCLUDED */
