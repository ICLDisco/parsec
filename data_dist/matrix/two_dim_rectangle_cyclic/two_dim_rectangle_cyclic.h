/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __TWO_DIM_RECTANGLE_CYCLIC_H__
#define __TWO_DIM_RECTANGLE_CYCLIC_H__

#ifdef USE_MPI
#include <mpi.h>
#endif /* USE_MPI */

#include "data_distribution.h"
#include "matrix.h"

/*
 * General distribution of data. Suppose exists a matrix in process of mpi rank 0
 */


/*******************************************************************
 * distributed data structure and basic functionalities
 *******************************************************************/

/* structure equivalent to PLASMA_desc, but for distributed matrix data
 */
typedef struct two_dim_block_cyclic {
    tiled_matrix_desc_t super;
    void *mat;          // pointer to the beginning of the matrix
    int nrst;           // max number of tile rows in a super-tile
    int ncst;           // max number of tile columns in a super tiles
    int GRIDrows;       // number of processes rows in the process grid
    int GRIDcols;       // number of processes cols in the process grid - derived parameter
    int colRANK;        // process column rank in the process grid - derived parameter
    int rowRANK;        // process row rank in the process grid - derived parameter
    int nb_elem_r;      // number of row of tiles  handled by this process - derived parameter
    int nb_elem_c;      // number of column of tiles handled by this process - derived parameter
} two_dim_block_cyclic_t;

/************************************************
 *   mpi ranks distribution in the process grid
 *   -----------------
 *   | 0 | 1 | 2 | 3 |
 *   |---------------|
 *   | 4 | 5 | 6 | 7 |
 *   -----------------
 ************************************************/

// #define A(m,n) &((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)]

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
 * @param nrst number of rows of tiles for block distribution
 * @param ncst number of column of tiles for block distribution
 * @param process_GridRows number of row of processes of the process grid (has to divide nodes)
 */
void two_dim_block_cyclic_init(two_dim_block_cyclic_t * Ddesc,enum matrix_type mtype, int nodes, int cores, int myrank, int mb, int nb, int ib, int lm, int ln, int i, int j, int m, int n, int nrst, int ncst, int process_GridRows );


#ifdef USE_MPI

int open_matrix_file(char * filename, MPI_File * handle, MPI_Comm comm);

int close_matrix_file(MPI_File * handle);

#endif /* USE_MPI */


#endif /* __TWO_DIM_RECTANGLE_CYCLIC_H__*/
