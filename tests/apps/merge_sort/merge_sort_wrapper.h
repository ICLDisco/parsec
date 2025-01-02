#if !defined(_MISSING_MERGE_SORT_WRAPPER_H_)
#define _MISSING_MERGE_SORT_WRAPPER_H_
/*
 * Copyright (c) 2014-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"

parsec_taskpool_t *merge_sort_new(parsec_tiled_matrix_t *A, int size, int nt);

#endif
