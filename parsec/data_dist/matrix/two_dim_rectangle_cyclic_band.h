/*
 * Copyright (c) 2018-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __TWO_DIM_RECTANGLE_CYCLIC_BAND_H__
#define __TWO_DIM_RECTANGLE_CYCLIC_BAND_H__
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/runtime.h"
#include "parsec/data.h"
#include <assert.h>

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

/* New structure */
typedef struct parsec_matrix_block_cyclic_band_s {
    parsec_tiled_matrix_t super;
    parsec_matrix_block_cyclic_t band;
    parsec_matrix_block_cyclic_t off_band;
    unsigned int band_size;     /** Number of band rows = 2 * band_size - 1 */
} parsec_matrix_block_cyclic_band_t;

/*
 * parsec_matrix_block_cyclic_band_t structure init
 * It inherits from off-band, so should be called after initialization of off_band
 */
void parsec_matrix_block_cyclic_band_init(
    parsec_matrix_block_cyclic_band_t *desc,
    int nodes, int myrank, int band_size );

/* include deprecated symbols */
#include "parsec/data_dist/matrix/deprecated/two_dim_rectangle_cyclic_band.h"

#endif // __TWO_DIM_RECTANGLE_CYCLIC_BAND_H__
