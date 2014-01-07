/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __SYM_TWO_DIM_RECTANGLE_CYCLIC_H__
#define __SYM_TWO_DIM_RECTANGLE_CYCLIC_H__

#include "data_dist/matrix/matrix.h"
#include "data_dist/matrix/grid_2Dcyclic.h"

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

/*******************************************************************
 * distributed data structure and basic functionalities
 *******************************************************************/

/* structure equivalent to PLASMA_desc, but for distributed matrix data
 */

/* to not rely on Plasma there, put our definition of upper/lower values (same as Plasma for compatibility) */
#define MatrixUpper         121
#define MatrixLower         122

typedef struct sym_two_dim_block_cyclic {
    tiled_matrix_desc_t super;
    grid_2Dcyclic_t grid;
    void *mat;              /**< pointer to the beginning of the matrix */
    int uplo;
} sym_two_dim_block_cyclic_t;

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
 * @param Ddesc matrix description structure, already allocated, that will be initialize
 * @param nodes number of nodes
 * @param cores number of cores per node
 * @param myrank rank of the local node (as of mpi rank)
 * @param mb number of row in a tile
 * @param nb number of column in a tile
 * @param lm number of rows of the entire matrix
 * @param ln number of column of the entire matrix
 * @param i starting row index for the computation on a submatrix
 * @param j starting column index for the computation on a submatrix
 * @param m number of rows of the entire submatrix
 * @param n numbr of column of the entire submatrix
 * @param process_GridRows number of row of processes of the process grid (has to divide nodes)
 * @param uplo upper or lower triangular part of the matrix is kept
 */
void sym_two_dim_block_cyclic_init( sym_two_dim_block_cyclic_t * Ddesc,
                                    enum matrix_type mtype,
                                    int nodes, int cores, int myrank,
                                    int mb, int nb, int lm, int ln,
                                    int i, int j, int m, int n,
                                    int process_GridRows, int uplo );

#endif /* __TWO_DIM_RECTANGLE_CYCLIC_H__*/
