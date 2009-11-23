/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __DATA_MANAGEMENT__
#define __DATA_MANAGEMENT__


#include "plasma.h"
/*
 * General distribution of data. Suppose exists a matrix in process of mpi rank 0
 */


/*******************************************************************
 * distributed data structure and basic functionalities
 *******************************************************************/

/* structure equivalent to PLASMA_desc, but for distributed matrix data
 */
typedef struct dplasma_desc_t {
    void *mat;          // pointer to the beginning of the matrix
    PLASMA_enum dtyp;   // precision of the matrix
    int mb;             // number of rows in a tile
    int nb;             // number of columns in a tile
    int bsiz;           // size in elements including padding
    int lm;             // number of rows of the entire matrix
    int ln;             // number of columns of the entire matrix
    int lmt;            // number of tile rows of the entire matrix - derived parameter
    int lnt;            // number of tile columns of the entire matrix - derived parameter
    int i;              // row index to the beginning of the submatrix
    int j;              // column indes to the beginning of the submatrix
    int m;              // number of rows of the submatrix
    int n;              // number of columns of the submatrix
    int mt;             // number of tile rows of the submatrix - derived parameter
    int nt;             // number of tile columns of the submatrix - derived parameter
    int mpi_rank;       // well... mpi rank...
    int GRIDrows;       // number of processes rows in the process grid
    int GRIDcols;       // number of processes cols in the process grid
    int colRANK;        // process column rank in the process grid - derived parameter
    int rowRANK;        // process row rank in the process grid - derived parameter
} DPLASMA_desc;

/************************************************
 *   mpi ranks distribution in the process grid
 *   -----------------
 *   | 0 | 1 | 2 | 3 |
 *   |----------------
 *   | 4 | 5 | 6 | 7 |
 *   -----------------
 ************************************************/

// #define A(m,n) &((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)]
/* initialize structure */
int dplasma_desc_init(PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc);

/* computing the mpi process rank that should handle tile A(m,n) */

int dplasma_get_rank_for_tile(DPLASMA_desc * Ddesc, int m, int n);

/* get a pointer to a specific tile handled locally
 *  return NULL if tile not a local tile
 */
void * dplasma_get_tile(DPLASMA_desc * Ddesc, int m, int n);

/* set new data to tile
 * return 0 if success, >0 if not
 */
int dplasma_set_tile(DPLASMA_desc * Ddesc, int m, int n, void * buff);


/****************************************************************
 * matrix generation, tiling and distribution
 ****************************************************************/

/* distribute the matrix to the different mpi ranks 
 * matrix -> pointer to matrix data on mpi rank 0 // NULL for other ranks
 */
int distribute_data(PLASMA_desc * Pdesc , DPLASMA_desc * Ddesc, MPI_Request ** reqs);


/* test if the matrix data has been distributed
 * return 0 if not
 */
int is_data_distributed(DPLASMA_desc * Ddesc, MPI_Request * reqs);


/* find which mpi rank handles a particular tile
 * row -> row of the data
 * col -> column of the data
 * return mpi rank
 */
int where_is_data(int row, int col);


/* generate a random matrix  */
int generate_matrix(int N, double * A1, double * A2, double * B1, double * B2, double * WORK, double * D,int LDA, int NRHS, int LDB);

/* convert to plasma desc and tiling format */
int tiling(PLASMA_enum * uplo, int N, double *A, int LDA, PLASMA_desc * descA);

/* debugging print of blocks */
void data_dist_verif(PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc );

#endif
