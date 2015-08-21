/*
 * Copyright (c) 2014-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _ep_wrapper_h
#define _ep_wrapper_h

#include "dague.h"
#include "dague/data_distribution.h"

/**
 * @param [IN] A     the data, already distributed and allocated
 * @param [IN] nt    number of tasks at a given level
 * @param [IN] level number of levels
 *
 * @return the dague object to schedule.
 */
dague_handle_t *ep_new(dague_ddesc_t *A, int nt, int level);

/**
 * @param [INOUT] o the dague object to destroy
 */
void ep_destroy(dague_handle_t *o);

#endif
