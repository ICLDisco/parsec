/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __TWO_DIM_RECTANGLE_CYCLIC_H__
#define __TWO_DIM_RECTANGLE_CYCLIC_H__

#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/grid_2Dcyclic.h"

BEGIN_C_DECLS

/*
 * General distribution of data. Suppose exists a matrix in process of mpi rank 0
 */

/*******************************************************************
 * distributed data structure and basic functionalities
 *******************************************************************/

/* structure equivalent to PLASMA_desc, but for distributed matrix data
 */
typedef struct parsec_matrix_block_cyclic {
    parsec_tiled_matrix_t super;
    parsec_grid_2Dcyclic_t   grid;
    void *mat;      /**< pointer to the beginning of the matrix */
    int nb_elem_r;  /**< number of row of tiles  handled by this process - derived parameter */
    int nb_elem_c;  /**< number of column of tiles handled by this process - derived parameter */
} parsec_matrix_block_cyclic_t;

/************************************************
 *   mpi ranks distribution in the process grid PxQ=2x4
 *   -----------------
 *   | 0 | 1 | 2 | 3 |
 *   |---------------|
 *   | 4 | 5 | 6 | 7 |
 *   -----------------
 ************************************************/

// #define A(m,n) &((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)]

/**
 * Initialize the description of a  2-D block cyclic distributed matrix.
 * @param dc matrix description structure, already allocated, that will be initialize
 * @param mtype type of data used for this matrix
 * @param storage type of storage of data
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
 * @param kp number of rows of tiles for k-cyclic block distribution
 *   act as-if the process grid had nkp repetitions for every process row
 *   (see example for kq below)
 * @param kq number of column of tiles for k-cyclic block distribution
 *   act as-if the process grid had kq repetitions for every process column
 *   For example, kp=1, kq=2 leads to the following pxq=2x4 process grid
 *   | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 |
 *   | 4 | 4 | 5 | 5 | 6 | 6 | 7 | 7 |
 * @param ip starting point on the process grid rows
 * @param jq starting point on the process grid cols
  */
void parsec_matrix_block_cyclic_init(parsec_matrix_block_cyclic_t * twoDBCdesc,
                               parsec_matrix_type_t mtype,
                               parsec_matrix_storage_t storage,
                               int myrank,
                               int mb,    int nb,   /* Tile size */
                               int lm,    int ln,   /* Global matrix size (what is stored)*/
                               int i,     int j,    /* Staring point in the global matrix */
                               int m,     int n,    /* Submatrix size (the one concerned by the computation */
                               int p,     int q,    /* process process grid*/
                               int kp,    int kq,   /* k-cyclicity */
                               int ip,    int jq);   /* starting point on the process grid*/

void parsec_matrix_block_cyclic_lapack_init(parsec_matrix_block_cyclic_t * twoDBCdesc,
                                      parsec_matrix_type_t mtype,
                                      parsec_matrix_storage_t storage,
                                      int myrank,
                                      int mb,   int nb,   /* Tile size */
                                      int lm,   int ln,   /* Global matrix size (what is stored)*/
                                      int i,    int j,    /* Staring point in the global matrix */
                                      int m,    int n,    /* Submatrix size (the one concerned by the computation */
                                      int p,     int q,   /* process process grid*/
                                      int kp,    int kq,  /* k-cyclicity */
                                      int ip,    int jq,  /* starting point on the process grid*/
                                      int mloc, int nloc);/* number of local rows and cols of the matrix */

/**
 * kcyclic _view_ of the 2-D Block cyclic distributed matrix. The goal is to
 * improve access locality by changing access order without incurring the cost of a physical
 * redistribution of the dataset. The underlying data storage is unchanged,
 * but the view provide accessors that swap lines and column blocks so that a block
 * from the same processor is provided for @kp repetitions along the m direction
 * (@kq along n, respectively); until such repetition is not possible anymore
 * (right edge, or bottom of the matrix where not enough local tiles are available).
 *
 * For example, starting from a standard 2D grid PxQ=2x2 with kp=2;
 * and m/mb = 6;
 *   rank_of(0,0) is 0; data_of(0,0) is origin (0,0)
 *   rank_of(1,0) is 0 (because kp=2); data_of(1,0) is origin (2,0)
 *   rank_of(2,0) is 1; data_of(2,0) is origin (1,0)
 *   rank_of(3,0) is 1 (because kp=2); data_of(3,0) is origin (3,0)
 *   rank_of(4,0) is 0; data_of(4,0) is origin (4,0)
 *   rank_of(5,0) is 1; data_of(5,0) is origin (5,0): despite kp=2,
 *     there are not enough local tiles on rank 0 to satisfy the view
 *     origin (6,0) does not exist), hence the view provides the next
 *     block on the m direction.
 *
 * Beware that using a kcyclic view is equivalent to swapping rows (or
 * columns, respectively), and the algorithm operating on the data may have
 * special requirements for it to be applicable. For example, a kcyclic
 * view of a diagonal dominant matrix is not always diagonal dominant; and the
 * result of a factorization on a kcyclic view is not a 'triangular' matrix,
 * but a swap, according to the view (i.e., solving the system requires applying
 * a compatible view on the right-hand side as well).
 *
 */
void parsec_matrix_block_cyclic_kview( parsec_matrix_block_cyclic_t* target,
                                 parsec_matrix_block_cyclic_t* origin,
                                 int kp, int kq );

/* include deprecated symbols */
#include "parsec/data_dist/matrix/deprecated/two_dim_rectangle_cyclic.h"

END_C_DECLS

#endif /* __TWO_DIM_RECTANGLE_CYCLIC_H__*/
