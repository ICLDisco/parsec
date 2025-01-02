/*
 * Copyright (c) 2013-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef branching_data_h
#define branching_data_h

#include "parsec/runtime.h"
#include "parsec/data.h"

parsec_data_collection_t *create_and_distribute_data(int rank, int world, int size, int nb);
void free_data(parsec_data_collection_t *d);

#endif
