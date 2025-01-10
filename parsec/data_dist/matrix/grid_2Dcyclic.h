/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __GRID_2DCYCLIC_H__
#define __GRID_2DCYCLIC_H__

#include "parsec/parsec_config.h"

BEGIN_C_DECLS

/*******************************************************************
 * 2D (or 1D) cyclic distribution
 *******************************************************************/

/* Placeholder for all relevant 2D distribution parameters */
typedef struct parsec_grid_2Dcyclic {
    int rank;       /**< Sequential rank of this processor */
    int rows;       /**< number of processes rows in the process grid */
    int cols;       /**< number of processes cols in the process grid - derived parameter */
    int ip;          /**< process row over which the first row of the array is distributed. */
    int jq;          /**< process column over which the first column of the array is distributed. */
    int krows;      /**< max number of tile rows in a k-cyclic distribution */
    int kcols;      /**< max number of tile columns in a k-cyclic distribution */
    int crank;      /**< process column rank in the process grid - derived parameter */
    int rrank;      /**< process row rank in the process grid - derived parameter */
    int rloc;       /**< number of row of tiles  handled by this process - derived parameter */
    int cloc;       /**< number of column of tiles handled by this process - derived parameter */
    int vp_p;       /**< number of rows used for data distribution by the VP */
    int vp_q;       /**< number of cols used for data distribution by the VP */
} parsec_grid_2Dcyclic_t;

/************************************************
 *   sequential ranks distribution
 *   in a 2x4 process grid
 *   -----------------
 *   | 0 | 1 | 2 | 3 |
 *   |---------------|
 *   | 4 | 5 | 6 | 7 |
 *   -----------------
 ************************************************/

/**
 * Initialize the description of a  2-D cyclic distribution.o
 * @param rank: sequential rank of the local node (as of mpi rank)
 * @param P: number of row of processes of the process grid
 * @param Q: number of colums of the processes of the process grid
 * @param kp: number of consecutive tiles along rows held by the same processor
 * @param kq: number of consecutive tiles along columns held by the same processor
 * @param ip: process row over which the first row of the array is distributed.
 * @param jq: process column over which the first column of the array is distributed
 *
 */
void parsec_grid_2Dcyclic_init(parsec_grid_2Dcyclic_t* grid, int rank, int P, int Q, int kp, int kq, int ip, int jq);

/* include deprecated symbols */
#include "parsec/data_dist/matrix/deprecated/grid_2Dcyclic.h"

END_C_DECLS

#endif /* __GRID_2DCYCLIC_H__*/
