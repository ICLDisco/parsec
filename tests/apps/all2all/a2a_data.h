#ifndef a2a_data_h
#define a2a_data_h
/*
 * Copyright (c) 2021-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "parsec/data_dist/matrix/matrix.h"

parsec_tiled_matrix_t *create_and_distribute_data(int rank, int world, int size);
void free_data(parsec_tiled_matrix_t *d);

#endif
