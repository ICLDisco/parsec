/*
 * Copyright (c) 2014-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef rtt_data_h
#define rtt_data_h

#include "parsec.h"
#include "parsec/data.h"

parsec_ddesc_t *create_and_distribute_data(int rank, int world, int size);
void free_data(parsec_ddesc_t *d);

#endif
