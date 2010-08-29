/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _DATA_DISTRIBUTION_H_ 
#define _DATA_DISTRIBUTION_H_ 

#include <stdarg.h>
#include <stdint.h>
#ifdef USE_MPI
#include "mpi.h"
#endif /*USE_MPI */

typedef struct dague_ddesc {
    uint32_t myrank;  /**< process rank */
    uint32_t cores;   /**< number of cores used for computation per node */
    uint32_t nodes;   /**< number of nodes involved in the computation */
    uint32_t (*rank_of)(struct dague_ddesc *mat, ...);
    void *   (*data_of)(struct dague_ddesc *mat, ...);
#ifdef USE_MPI
    MPI_Comm comm;
#endif /* USE_MPI */
} dague_ddesc_t;

#endif /* _DATA_DISTRIBUTION_H_ */

