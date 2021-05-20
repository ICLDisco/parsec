/*
 * Copyright (c) 2017-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "parsec/data_dist/matrix/matrix.h"

#if defined(PARSEC_HAVE_MPI)
#include <mpi.h>
#endif

/**
 * @param [in] Y:    the data, already distributed and allocated
 * @param [in] uplo: Upper / Lower / UpperLower 
 * @return the parsec object to schedule.
 */
parsec_taskpool_t* 
parsec_two_dim_band_New(parsec_tiled_matrix_dc_t *Y, int uplo);

/**
 * @param [inout] the parsec object to destroy
 */
void parsec_two_dim_band_Destruct(parsec_taskpool_t *taskpool);

/**
 * @brief Init dcY
 *
 * @param [inout] dcY: the data, already distributed and allocated
 * @param [in] uplo: Upper / Lower / UpperLower 
 */
int parsec_two_dim_band_test(parsec_context_t *parsec,
                        parsec_tiled_matrix_dc_t *dcY, int uplo);

/**
 * @param [in] Y:    the data, already distributed and allocated
 * @param [in] uplo: Upper / Lower / UpperLower 
 * @return the parsec object to schedule.
 */
parsec_taskpool_t*
parsec_two_dim_band_free_New(parsec_tiled_matrix_dc_t *Y, int uplo);

/**
 * @param [inout] the parsec object to destroy
 */
void parsec_two_dim_band_free_Destruct(parsec_taskpool_t *taskpool);

/**
 * @brief Free dcY
 * 
 * @param [inout] dcY: the data, already distributed and allocated
 * @param [in] uplo: Upper / Lower / UpperLower 
 */
int parsec_two_dim_band_free(parsec_context_t *parsec,
                            parsec_tiled_matrix_dc_t *dcY, int uplo);
