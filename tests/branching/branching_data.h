/*
 * Copyright (c) 2013-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef branching_data_h
#define branching_data_h

#include "dague.h"
#include "dague/types.h"

dague_ddesc_t *create_and_distribute_data(int rank, int world, int size);
void free_data(dague_ddesc_t *d);

#endif
