/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#if !defined(_MISSING_BT_REDUCTION_WRAPPER_H_)
#define _MISSING_BT_REDUCTION_WRAPPER_H_

#include "parsec/runtime.h"
#include "parsec/data_distribution.h"
#include "parsec/data_dist/matrix/matrix.h"

parsec_taskpool_t *BT_reduction_new(parsec_tiled_matrix_t *A, int size, int nt);

#endif
