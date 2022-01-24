/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef __SYM_TWO_DIM_RECTANGLE_CYCLIC_H__
#error "Deprecated headers must not be included directly!"
#endif // __SYM_TWO_DIM_RECTANGLE_CYCLIC_H__

/*
 * Symmetrical matrix. 2D block cyclic distribution, lower tiles distributed only
 *
 * --
 *|0 |
 * --|--
 *|2 |3 |
 *|--|--|--
 *|0 |1 |0 |
 *|--|--|--|--
 *|2 |3 |2 |3 |
 * -----------
 *
 */

typedef parsec_matrix_sym_block_cyclic_t sym_two_dim_block_cyclic_t __parsec_attribute_deprecated__("Use parsec_matrix_sym_block_cyclic_t");

/************************************************
 *   mpi ranks distribution in the process grid
 *   -----------------
 *   | 0 | 1 | 2 | 3 |
 *   |---------------|
 *   | 4 | 5 | 6 | 7 |
 *   -----------------
 ************************************************/


/**
 * Initialize the description of a  2-D block cyclic distributed matrix.
 * @param dc matrix description structure, already allocated, that will be initialize
 * @param nodes number of nodes
 * @param myrank rank of the local node (as of mpi rank)
 * @param mb number of row in a tile
 * @param nb number of column in a tile
 * @param lm number of rows of the entire matrix
 * @param ln number of column of the entire matrix
 * @param i starting row index for the computation on a submatrix
 * @param j starting column index for the computation on a submatrix
 * @param m number of rows of the entire submatrix
 * @param n numbr of column of the entire submatrix
 * @param p number of row of processes of the process grid the
 *   resulting distribution will be made so that pxq=nodes
 * @param q number of col of processes of the process grid the
 *   resulting distribution will be made so that pxq=nodes
 * @param uplo upper or lower triangular part of the matrix is kept
 */
static inline
void sym_two_dim_block_cyclic_init( parsec_matrix_sym_block_cyclic_t * dc,
                                    parsec_matrix_type_t mtype,
                                    int myrank,
                                    int mb, int nb, int lm, int ln,
                                    int i, int j, int m, int n,
                                    int P, int Q, /* process process grid */
                                    parsec_matrix_uplo_t uplo )
    __parsec_attribute_deprecated__("Use parsec_matrix_sym_block_cyclic_init");

static inline
void sym_two_dim_block_cyclic_init( parsec_matrix_sym_block_cyclic_t * dc,
                                    parsec_matrix_type_t mtype,
                                    int myrank,
                                    int mb, int nb, int lm, int ln,
                                    int i, int j, int m, int n,
                                    int P, int Q, /* process process grid */
                                    parsec_matrix_uplo_t uplo )
{
    parsec_matrix_sym_block_cyclic_init(dc, mtype, myrank, mb, nb, lm, ln, i, j, m, n, P, Q, uplo);
}
