/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __TABULAR_DISTRIBUTION_H__
#define __TABULAR_DISTRIBUTION_H__

#ifdef HAVE_MPI
#include <mpi.h>
#endif /* HAVE_MPI */

#include "data_dist/matrix/matrix.h"

/*
 * General distribution of data. Suppose exists a matrix in process of mpi rank 0
 */
struct dague_data_s;

/*******************************************************************
 * distributed data structure and basic functionalities
 *******************************************************************/

typedef struct tile_elem {
    uint32_t             rank;
    int32_t              vpid;
    struct dague_data_s* data;
    void*                tile;
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
 * @param lm number of rows of the entire matrix
 * @param ln number of column of the entire matrix
 * @param i starting row index for the computation on a submatrix
 * @param j starting column index for the computation on a submatrix
 * @param m number of rows of the entire submatrix
 * @param n numbr of column of the entire submatrix
 * @param table mpi rank for each tile, column major ordering
 */

void tabular_distribution_init(tabular_distribution_t* Ddesc,
                               enum matrix_type mtype,
                               unsigned int nodes,
                               unsigned int cores,
                               unsigned int myrank,
                               unsigned int mb,
                               unsigned int nb,
                               unsigned int lm,
                               unsigned int ln,
                               unsigned int i,
                               unsigned int j,
                               unsigned int m,
                               unsigned int n,
                               unsigned int* table);




unsigned int* create_2dbc(unsigned int size, unsigned int block, unsigned int nbproc, unsigned int Grow);

#endif /* __TABULAR_DISTRIBUTION_H__ */
