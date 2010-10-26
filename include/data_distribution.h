/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _DATA_DISTRIBUTION_H_ 
#define _DATA_DISTRIBUTION_H_ 

#include "dague_config.h"

#include <stdarg.h>
#include <unistd.h>
#include <stdint.h>
#ifdef USE_MPI
#include "mpi.h"
#endif /*USE_MPI */

typedef struct dague_ddesc {
    uint32_t myrank;  /**< process rank */
    uint32_t cores;   /**< number of cores used for computation per node */
    uint32_t nodes;   /**< number of nodes involved in the computation */
    uint32_t (*rank_of)(struct dague_ddesc *mat, ...); /* return the rank of the process owning the data  */
    void *   (*data_of)(struct dague_ddesc *mat, ...); /* return the pointer to the data pocessed locally */
#ifdef DAGUE_PROFILING
    uint32_t (*data_key)(struct dague_ddesc *mat, ...); /* return a unique key (unique only for the specified dague_ddesc) associated to a data */
    int (*key_to_string)(struct dague_ddesc *mat, uint32_t datakey, char * buffer, uint32_t buffer_size); /* compute a string in 'buffer' meaningful for profiling about data, return the size of the string */
    char      *key_dim;
    char      *key;
#endif /* DAGUE_PROFILING */
#ifdef USE_MPI
    MPI_Comm comm;
#endif /* USE_MPI */
} dague_ddesc_t;

#endif /* _DATA_DISTRIBUTION_H_ */

