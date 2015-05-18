/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef schedmicro_data_h
#define schedmicro_data_h

#include "dague.h"

dague_ddesc_t *create_and_distribute_data(int rank, int world, int size, int seg);
void free_data(dague_ddesc_t *d);

#endif
