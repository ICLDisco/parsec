/*
 * Copyright (c) 2009-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef choice_data_h
#define choice_data_h

#include "parsec/runtime.h"
#include "parsec/data.h"

parsec_data_collection_t *create_and_distribute_data(int rank, int world, int size);
void free_data(parsec_data_collection_t *d);

#endif
