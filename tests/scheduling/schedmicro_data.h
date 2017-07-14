/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef schedmicro_data_h
#define schedmicro_data_h

#include "parsec.h"
#include "parsec/data.h"

parsec_data_collection_t *create_and_distribute_data(int rank, int world, int size, int seg);
void free_data(parsec_data_collection_t *d);

#endif
