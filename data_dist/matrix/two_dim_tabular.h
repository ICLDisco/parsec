/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __TWODTD_H__
#define __TWODTD_H__

#ifdef HAVE_MPI
#include <mpi.h>
#endif /* HAVE_MPI */

#include "data_dist/matrix/matrix.h"

/*
 * General distribution of data.
 */


/*******************************************************************
 * distributed data structure and basic functionalities
 *******************************************************************/

typedef struct two_dim_td_table_elem_s {
    uint32_t rank;
    int32_t vpid;
    void * tile;
} two_dim_td_table_elem_t;

typedef struct two_dim_td_table_s {
    int nbelem;
    two_dim_td_table_elem_t elems[1]; /**< Elements of table are arranged column major. */
} two_dim_td_table_t;

/* structure equivalent to PLASMA_desc, but for distributed matrix data
 */
typedef struct two_dim_tabular_s {
    tiled_matrix_desc_t super;
    two_dim_td_table_t *tiles_table;
} two_dim_tabular_t;

/**
 * Initialize the description of a tabular abribtrary distribution
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
 * @param table associative table to assign tiles to all ranks. Can be NULL.
 *        In that case, you need to call set_table or set_random_table before
 *        using that descriptor.
 */

void two_dim_tabular_init(two_dim_tabular_t * Ddesc,
                          enum matrix_type mtype,
                          unsigned int nodes, unsigned int cores, unsigned int myrank,
                          unsigned int mb, unsigned int nb,
                          unsigned int lm, unsigned int ln,
                          unsigned int i, unsigned int j,
                          unsigned int m, unsigned int n,
                          two_dim_td_table_t *table );

void two_dim_tabular_set_table(two_dim_tabular_t *Ddesc, two_dim_td_table_t *table);
void two_dim_tabular_set_random_table(two_dim_tabular_t *Ddesc, unsigned int seed);
void two_dim_td_table_clone_table_structure(two_dim_tabular_t *Src, two_dim_tabular_t *Dst);

void two_dim_tabular_free_table(two_dim_td_table_t *table);

#endif /* __TWODTD_H__ */
