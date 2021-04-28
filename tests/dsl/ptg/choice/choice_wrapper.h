/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _choice_wrapper_h
#define _choice_wrapper_h

#include "parsec/runtime.h"
#include "parsec/data.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the parsec object to schedule.
 */
parsec_taskpool_t *choice_new(parsec_data_collection_t *A, int size, int *decision, int nb, int world);

/**
 * @param [INOUT] o the parsec object to destroy
 */
void choice_destroy(parsec_taskpool_t *o);

#endif 
