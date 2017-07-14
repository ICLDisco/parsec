/*
 * Copyright (c) 2014-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _rtt_wrapper_h
#define _rtt_wrapper_h

#include "parsec.h"
#include "parsec/data_distribution.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element (in bytes)
 * @param [IN] nb   number of iterations
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t *rtt_new(parsec_data_collection_t *A, int size, int nb);

/**
 * @param [INOUT] o the parsec object to destroy
 */
void rtt_destroy(parsec_taskpool_t *o);

#endif 
