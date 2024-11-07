/*
 * Copyright (c) 2023-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/runtime.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

parsec_taskpool_t *check_multisize_bcast_new(parsec_matrix_block_cyclic_t *A, int size, int nt);

