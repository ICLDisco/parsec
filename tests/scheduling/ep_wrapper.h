/*
 * Copyright (c) 2014-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _ep_wrapper_h
#define _ep_wrapper_h

#include "parsec.h"
#include "parsec/data_distribution.h"

/**
 * @param [IN] A     the data, already distributed and allocated
 * @param [IN] nt    number of tasks at a given level
 * @param [IN] level number of levels
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t *ep_new(parsec_data_collection_t *A, int nt, int level);

/**
 * @param [INOUT] o the parsec object to destroy
 */
void ep_destroy(parsec_taskpool_t *o);

#endif
