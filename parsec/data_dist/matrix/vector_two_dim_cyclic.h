/*
 * Copyright (c) 2009-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#ifndef __VECTOR_TWO_DIM_CYCLIC_H__
#define __VECTOR_TWO_DIM_CYCLIC_H__

#include "parsec/data_dist/matrix/matrix.h"
#include "parsec/data_dist/matrix/grid_2Dcyclic.h"

BEGIN_C_DECLS

/*******************************************************************
 * distributed data vector and basic functionalities
 *******************************************************************/
typedef enum parsec_vector_two_dim_cyclic_distrib_t {
    PARSEC_VECTOR_DISTRIB_ROW,
    PARSEC_VECTOR_DISTRIB_COL,
    PARSEC_VECTOR_DISTRIB_DIAG
} parsec_vector_two_dim_cyclic_distrib_t;

/*
 * Vector structure inheriting from parsec_matrix_t
 * Follows the same distribution than the diagonal tiles of the
 * parsec_matrix_block_cyclic_t structure.
 */
typedef struct parsec_vector_two_dim_cyclic_t_s {
    parsec_tiled_matrix_t super;
    parsec_grid_2Dcyclic_t     grid;
    parsec_vector_two_dim_cyclic_distrib_t    distrib; /**< Distribution used for the vector: Row, Column or diagonal */
    int   lcm;                   /**< number of processors present on diagonal */
    void *mat;                   /**< pointer to the beginning of the matrix   */
} parsec_vector_two_dim_cyclic_t;

/**
 * Initialize the description of a 2-D block cyclic distributed vector.
 *
 * @param dc matrix description structure, already allocated, that will be initialize
 * @param mtype type of data used for this matrix
 * @param myrank rank of the local node (as of mpi rank)
 * @param mb number of elements in a segment
 * @param lm number of elements in the full vector
 * @param i starting element index for the computation on a subvector
 * @param m number of elements of the entire subvector
 * @param p number of row of processes of the process grid the
 *   resulting distribution will be made so that pxq=nodes
 * @param q number of col of processes of the process grid the
 *   resulting distribution will be made so that pxq=nodes
 */
void parsec_vector_two_dim_cyclic_init(parsec_vector_two_dim_cyclic_t * vdesc,
                                         parsec_matrix_type_t    mtype,
                                         enum parsec_vector_two_dim_cyclic_distrib_t distrib,
                                         int myrank,
                                         int mb, int lm, int i, int m,
                                         int P, int Q );

/* include deprecated symbols */
#include "parsec/data_dist/matrix/deprecated/vector_two_dim_cyclic.h"

END_C_DECLS

#endif /* __VECTOR_TWO_DIM_CYCLIC_H__*/
