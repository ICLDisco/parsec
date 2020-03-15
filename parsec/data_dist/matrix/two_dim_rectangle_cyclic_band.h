/*
 * Copyright (c) 2018-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/runtime.h"
#include "parsec/data.h"
#include <assert.h>

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

/* New structure */
typedef struct two_dim_block_cyclic_band_s {
    parsec_tiled_matrix_dc_t super;
    two_dim_block_cyclic_t band;
    two_dim_block_cyclic_t off_band;
    unsigned int band_size;     /** Number of band rows = 2 * band_size - 1 */ 
} two_dim_block_cyclic_band_t;

/* 
 * two_dim_block_cyclic_band_t structure init 
 * It inherits from off-band, so should be called after initialization of off_band
 */
void two_dim_block_cyclic_band_init( two_dim_block_cyclic_band_t *desc,
                                     int nodes, int myrank, int band_size );
