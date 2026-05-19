/*
 * Copyright (c) 2023      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 */

#ifndef __PARSEC_MATRIX_SBC_H__
#define __PARSEC_MATRIX_SBC_H__

#include <stdint.h>

#include "parsec/data_dist/matrix/matrix.h"

BEGIN_C_DECLS

/**
 * Descriptor for the Symmetric Block-Cyclic (SBC) distribution.
 *
 * The matrix is tiled and only the triangular part selected by @ref uplo is
 * stored.  Tile ownership follows a repeated r x r periodic pattern.  Each
 * off-diagonal pattern pair (a,b)/(b,a) is mapped to the same rank, while
 * diagonal entries are assigned according to either the extended SBC pattern
 * (r*(r-1)/2 ranks) or the basic even-r pattern (r*r/2 ranks).
 */
typedef struct parsec_matrix_sbc_s {
    parsec_tiled_matrix_t super;
    void *mat;              /**< pointer to the beginning of the matrix */
    parsec_matrix_uplo_t uplo;
    uint16_t r;             /**< size of the repeated r x r SBC pattern */
    uint16_t diag_patterns; /**< number of diagonal patterns in use */
    uint8_t extended;       /**< true for the extended diagonal allocation */
} parsec_matrix_sbc_t;

/**
 * Initialize a symmetric block-cyclic distributed tiled matrix descriptor.
 *
 * The descriptor uses tile storage and packs each local rank's owned triangular
 * tiles contiguously.  Passing nodes == r*(r-1)/2 selects the extended SBC
 * variant from the paper.  Passing nodes == r*r/2 selects the basic variant,
 * which is valid only when r is even.
 *
 * @param dc matrix description structure, already allocated, that will be initialized
 * @param mtype type of matrix elements
 * @param myrank rank of the local node (as of mpi rank)
 * @param mb number of row in a tile
 * @param nb number of column in a tile
 * @param lm number of rows of the entire matrix
 * @param ln number of column of the entire matrix
 * @param i starting row index for the computation on a submatrix
 * @param j starting column index for the computation on a submatrix
 * @param m number of rows of the entire submatrix
 * @param n number of columns of the entire submatrix
 * @param nodes number of ranks participating in the distribution. The
 *   extended SBC variant uses r*(r-1)/2 ranks. The basic SBC variant uses
 *   r*r/2 ranks and is valid only for even r.
 * @param r size of the repeated r x r SBC pattern
 * @param uplo upper or lower triangular part of the matrix is kept
 */
int parsec_matrix_sbc_init( parsec_matrix_sbc_t * dc,
                            parsec_matrix_type_t mtype,
                            int myrank,
                            int mb, int nb, int lm, int ln,
                            int i, int j, int m, int n,
                            int nodes, int r,
                            parsec_matrix_uplo_t uplo );

END_C_DECLS

#endif /* __PARSEC_MATRIX_SBC_H__*/
