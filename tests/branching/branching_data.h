/*
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef branching_data_h
#define branching_data_h

#include "parsec.h"
#include "parsec/types.h"

parsec_ddesc_t *create_and_distribute_data(int rank, int world, int size);
void free_data(parsec_ddesc_t *d);

#endif
