/*
 * Copyright (c) 2014-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef rtt_data_h
#define rtt_data_h

#include "parsec/runtime.h"
#include "parsec/data.h"

parsec_data_collection_t *create_and_distribute_data(int rank, int world, int size);
void free_data(parsec_data_collection_t *d);

#endif
