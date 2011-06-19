/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 */
#ifndef _DPLASMA_H_
#define _DPLASMA_H_

#include "dague_config.h"

#define dplasma_error(__func, __msg) fprintf(stderr, "%s: %s\n", (__func), (__msg))

#include "data_dist/matrix/matrix.h"

#include "dplasma/include/dplasma_s.h"
#include "dplasma/include/dplasma_d.h"
#include "dplasma/include/dplasma_c.h"
#include "dplasma/include/dplasma_z.h"

#define DPLASMA_FLAT_TREE       0
#define DPLASMA_GREEDY_TREE     1
#define DPLASMA_FIBONACCI_TREE  2
#define DPLASMA_BINARY_TREE     3

void dplasma_pivgen( int type, tiled_matrix_desc_t *A, int *ipiv );

#endif /* _DPLASMA_H_ */
