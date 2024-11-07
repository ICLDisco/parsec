/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef _a2a_wrapper_h
#define _a2a_wrapper_h

#include "parsec/runtime.h"
#include "parsec/data_dist/matrix/matrix.h"

/**
 * @param [IN] A    the data, already distributed and allocated
 * @param [IN] size size of each local data element
 * @param [IN] nb   number of iterations
 *
 * @return the parsec handle to schedule.
 */
parsec_taskpool_t *a2a_new(parsec_tiled_matrix_t *A, parsec_tiled_matrix_t *B, int size, int repeat);

#endif
