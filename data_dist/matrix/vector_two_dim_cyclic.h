/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __VECTOR_TWO_DIM_CYCLIC_H__
#define __VECTOR_TWO_DIM_CYCLIC_H__

#ifdef HAVE_MPI
#include <mpi.h>
#endif /* HAVE_MPI */

#include "dague_config.h"
#include "data_dist/matrix/matrix.h"
#include "data_dist/matrix/grid_2Dcyclic.h"

/*
 * General distribution of data. Suppose exists a matrix in process of mpi rank 0
 */

/*******************************************************************
 * distributed data structure and basic functionalities
 *******************************************************************/

/* structure equivalent to PLASMA_desc, but for distributed matrix data
 */
typedef struct vector_two_dim_cyclic {
    tiled_matrix_desc_t super;
    grid_2Dcyclic_t     grid;
    void *mat;      /**< pointer to the beginning of the matrix */
    int nb_elem_r;  /**< number of row of tiles  handled by this process - derived parameter */
} vector_two_dim_cyclic_t;

/**
 * Initialize the description of a  2-D block cyclic distributed matrix.
 * @param Ddesc matrix description structure, already allocated, that will be initialize
 * @param mtype type of data used for this matrix
 * @param storage type of storage of data
 * @param nodes number of nodes
 * @param cores number of cores per node
 * @param myrank rank of the local node (as of mpi rank)
 * @param mb number of row in a tile
 * @param lm number of rows of the entire matrix
 * @param i starting row index for the computation on a submatrix
 * @param m number of rows of the entire submatrix
 * @param nrst number of rows of tiles for block distribution
 * @param process_GridRows number of row of processes of the process grid (has to divide nodes)
 */
void vector_two_dim_cyclic_init(vector_two_dim_cyclic_t * twoDBCdesc,
                               enum matrix_type mtype,
                               enum matrix_storage storage,
                               int nodes, int cores, int myrank,
                               int mb,   /* Tile size */
                               int lm,   /* Global matrix size (what is stored)*/
                               int i,    /* Staring point in the global matrix */
                               int m,    /* Submatrix size (the one concerned by the computation */
                               int nrst, /* Super-tiling size */
                               int process_GridRows );

void vector_two_dim_cyclic_supertiled_view( vector_two_dim_cyclic_t* target,
                                            vector_two_dim_cyclic_t* origin,
                                            int rst );

#endif /* __VECTOR_TWO_DIM_CYCLIC_H__*/
