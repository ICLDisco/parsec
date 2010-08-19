/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __TABULAR_DISTRIBUTION_H__
#define __TABULAR_DISTRIBUTION_H__

#ifdef USE_MPI
#include <mpi.h>
#endif /* USE_MPI */

#include "../../data_distribution.h"
#include "../matrix.h"

/*
 * General distribution of data. Suppose exists a matrix in process of mpi rank 0
 */


/*******************************************************************
 * distributed data structure and basic functionalities
 *******************************************************************/

typedef struct tile_elem {
    uint32_t rank;
    void * tile;
} tile_elem_t;

/* structure equivalent to PLASMA_desc, but for distributed matrix data
 */
typedef struct tabular_distribution {
    tiled_matrix_desc_t super;
    tile_elem_t * tiles_table;
} tabular_distribution_t;



/************************************************
 *   mpi ranks distribution in the process grid
 *         ------------------------------------------
 * tile    | 0,0 | 1,0 | 2,0 | 0,1 | 1,1 | 2,1 | ... |
 *         |-----------------------------------------|
 * process |  4  |  2  |  0  |  4  |  3  |  0  | ... | <-- input table for init
 *         ------------------------------------------
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
 * @param table mpi rank for each tile, column major ordering
 */

void tabular_distribution_init(tabular_distribution_t * Ddesc, enum matrix_type mtype, uint32_t nodes, uint32_t cores, uint32_t myrank, uint32_t mb, uint32_t nb, uint32_t ib, uint32_t lm, uint32_t ln, uint32_t i, uint32_t j, uint32_t m, uint32_t n, uint32_t * table );

#endif /* __TABULAR_DISTRIBUTION_H__ */
