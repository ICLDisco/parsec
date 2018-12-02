/*
 * Copyright (c) 2018-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/sym_two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/runtime.h"
#include "parsec/data.h"
#include <assert.h>

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

/* New structure */
typedef struct sym_two_dim_block_cyclic_band_s {
    sym_two_dim_block_cyclic_t super;
    two_dim_block_cyclic_t band;
    unsigned int band_size;     /** Number of band rows = band_size */
} sym_two_dim_block_cyclic_band_t;

/* New rank_of, rank_of_key for sym two dim block cyclic band */
uint32_t sym_twoDBC_band_rank_of(parsec_data_collection_t * desc, ...);
uint32_t sym_twoDBC_band_rank_of_key(parsec_data_collection_t *desc, parsec_data_key_t key);

/* New data_of, data_of_key for sym two dim block cyclic band */
parsec_data_t* sym_twoDBC_band_data_of(parsec_data_collection_t *desc, ...);
parsec_data_t* sym_twoDBC_band_data_of_key(parsec_data_collection_t *desc, parsec_data_key_t key);
