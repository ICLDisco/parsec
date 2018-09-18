/*
 * Copyright (c) 2011-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 */

#ifndef _DPLASMAAUX_H_INCLUDED
#define _DPLASMAAUX_H_INCLUDED

/**
 * Returns the priority limit of a specific function for a specific precision.
 *
 * @details
 *   This auxiliary helper function uses the process environment to
 *   find what priority limit the user wants to set for a given
 *   kernel.
 *
 *   The priority limit defines how priorities should pursue the
 *   critical path (see e.g. zpotrf_wrapper.c). By convention,
 *   the environment variable name is a capital S,D,C, or Z to
 *   specify the precision concatenated with the kernel name
 *   in capital (e.g. SPOTRF to set the priority limit of potrf
 *   in simple precision).
 *
 *    @param[IN] function: the base function name used to compose
 *                         the environment variable name
 *    @param[IN] dc:       a data collection used to find the precision
 *    @return the priority limit that the user wants to define for
 *            this function and precision.
 */
int dplasma_aux_get_priority_limit( char* function, const parsec_tiled_matrix_dc_t* dc );

/**
 * Returns the lookahead to use for GEMM
 *
 * @details
 *   This auxiliary helper function apply internal heuristics
 *   to determine the look ahead used in some SUMMA GEMM algorithms
 *   used in dplasma.
 *
 *  @param[IN] the data collection pointing to the A matrix of the
 *             GEMM operation.
 *  @return depending on the number of nodes and the matrix size,
 *          the value to use for the look ahead in SUMMA.
 */
int dplasma_aux_getGEMMLookahead( parsec_tiled_matrix_dc_t *A );

/**
 *  Create a dplasma-specific communicator
 *
 *  @details
 *    This function allocates a dplasma-specific communicator when
 *    compiling and running in distributed using MPI. It requires
 *    PaRSEC to be compiled with MPI.
 *
 *    Allocating a dplasma-specific communicator ensures that no
 *    dplasma communication in Wrappers will interfere with an
 *    application communication. It is recommended to use in
 *    MPI setups.
 *
 *    To ensure portability, the communicator is passed as a pointer,
 *    but it should point to an actual communicator that will be dupplicated.
 *
 *    If this function is not called, dplasma will use MPI_COMM_WORLD
 *    by default, creating risks of deadlocks and errors if some
 *    dplasma communication overlaps with other application communications
 *    on the same communicator.
 *
 *    dplasma_aux_free_comm must be called before MPI_Finalize() if
 *    dplasma_aux_dup_comm is called.
 *
 *    It is incorrect to call this function twice without calling
 *    dplasma_aux_free_comm between each call.
 *
 *    @param[IN] _psrc: a pointer to a valid MPI communicator
 *    @return the error code of MPI_Comm_dup
 */
int dplasma_aux_dup_comm(void *_psrc);

/**
 *  Free the dplasma-specific communicator
 *
 *  @details
 *     This function frees the communicator allocated via dplasma_aux_dup_comm.
 *
 *     @return the error code of MPI_Comm_free
 */
int dplasma_aux_free_comm(void);

/**
 * Globally visible pointer to the dplasma-specific communicator
 */
extern void *dplasma_pcomm;

#endif

