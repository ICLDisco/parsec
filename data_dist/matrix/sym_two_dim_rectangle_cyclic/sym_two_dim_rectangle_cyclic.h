/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __SYM_TWO_DIM_RECTANGLE_CYCLIC_H__
#define __SYM_TWO_DIM_RECTANGLE_CYCLIC_H__

#include "data_dist/matrix/matrix.h"

/*
 * Symmetrical matrix. 2D block cyclic distribution, lower tiles dsitributed only
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
typedef struct sym_two_dim_block_cyclic {
    tiled_matrix_desc_t super;
    void *mat;              /**< pointer to the beginning of the matrix */
    unsigned int GRIDrows;  /**< number of processes rows in the process grid */
    unsigned int GRIDcols;  /**< number of processes cols in the process grid - derived parameter */
    unsigned int colRANK;   /**< process column rank in the process grid - derived parameter */
    unsigned int rowRANK;   /**< process row rank in the process grid - derived parameter */
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
 * @param ib number of column in an inner block
 * @param lm number of rows of the entire matrix
 * @param ln number of column of the entire matrix
 * @param i starting row index for the computation on a submatrix
 * @param j starting column index for the computation on a submatrix
 * @param m number of rows of the entire submatrix
 * @param n numbr of column of the entire submatrix
 * @param process_GridRows number of row of processes of the process grid (has to divide nodes)
 */
void sym_two_dim_block_cyclic_init(sym_two_dim_block_cyclic_t * Ddesc,enum matrix_type mtype, unsigned int nodes, unsigned int cores, unsigned int myrank, unsigned int mb, unsigned int nb, unsigned int ib, unsigned int lm, unsigned int ln, unsigned int i, unsigned int j, unsigned int m, unsigned int n, unsigned int process_GridRows );


#endif /* __TWO_DIM_RECTANGLE_CYCLIC_H__*/
