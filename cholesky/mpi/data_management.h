/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __DATA_MANAGEMENT__
#define __DATA_MANAGEMENT__


#include "plasma.h"
#if defined(USE_MPI)
#include <mpi.h>
#endif  /* defined(USE_MPI) */
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
    int nrst;           // max number of tile rows in a super-tile
    int ncst;           // max number of tile columns in a super tiles
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
 *   |---------------|
 *   | 4 | 5 | 6 | 7 |
 *   -----------------
 ************************************************/

// #define A(m,n) &((double*)descA.mat)[descA.bsiz*(m)+descA.bsiz*descA.lmt*(n)]
/* initialize Ddesc from Pdesc */
int dplasma_desc_init(const PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc);

/* initialize and bcast Ddesc from Pdesc */
int dplasma_desc_bcast(const PLASMA_desc * Pdesc, DPLASMA_desc *Ddesc);

/* computing the mpi process rank that should handle tile A(m,n) */

int dplasma_get_rank_for_tile(DPLASMA_desc * Ddesc, int m, int n);

/* get a pointer to a specific tile 
 * if the tile is remote, it is downloaded first */
void * dplasma_get_tile(DPLASMA_desc * Ddesc, int m, int n);

/* get a pointer to a specific LOCAL tile */
void * dplasma_get_local_tile(DPLASMA_desc * Ddesc, int m, int n);


/* set new data to tile
 * return 0 if success, >0 if not
 */
int dplasma_set_tile(DPLASMA_desc * Ddesc, int m, int n, void * buff);


/****************************************************************
 * matrix generation, tiling and distribution
 ****************************************************************/
#if !defined(USE_MPI)
typedef struct MPI_Request MPI_Request;
#endif  /* !defined(USE_MPI) */
/* distribute the matrix to the different mpi ranks 
 * matrix -> pointer to matrix data on mpi rank 0 // NULL for other ranks
 */
int distribute_data(PLASMA_desc * Pdesc , DPLASMA_desc * Ddesc, MPI_Request ** reqs, int * req_count);

/* regroup the distributed tiles to the rank 0 */
/* Pdesc is NULL except on rank 0 */
int gather_data(PLASMA_desc * Pdesc, DPLASMA_desc *Ddesc);

/* test if the matrix data has been distributed
 * return 0 if not
 */
int is_data_distributed(DPLASMA_desc * Ddesc, MPI_Request * reqs, int req_count);


/* generate a random matrix  */
int generate_matrix(int N, double * A1, double * A2, double * B1, double * B2, double * WORK, double * D,int LDA, int NRHS, int LDB);

/* convert to plasma desc and tiling format */
int tiling(PLASMA_enum * uplo, int N, double *A, int LDA, PLASMA_desc * descA);
int untiling(PLASMA_enum * uplo, int N, double *A, int LDA, PLASMA_desc * descA);


/* debugging print of blocks */
void data_dist_verif(PLASMA_desc * Pdesc, DPLASMA_desc * Ddesc );
int data_dump(DPLASMA_desc * Ddesc);
int plasma_dump(PLASMA_desc * Pdesc);

#endif
