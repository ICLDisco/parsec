#if !defined(_DATA_GEN_H_)
#define _DATA_GEN_H_
/*
 * Copyright (c) 2023-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

parsec_matrix_block_cyclic_t *create_and_distribute_data(int rank, int world, int nb, int nt);
void free_data(parsec_matrix_block_cyclic_t *d);

#endif
