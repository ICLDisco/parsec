/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _DATA_DISTRIBUTION_H_ 
#define _DATA_DISTRIBUTION_H_ 

#include "dague_config.h"
#include "profiling.h"

#if defined(HAVE_STDARG_H)
#include <stdarg.h>
#endif  /* defined(HAVE_STDARG_H) */
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif  /* defined(HAVE_UNISTD_H) */
#include <stdint.h>
#include <stdlib.h>

#ifdef HAVE_MPI
#include "mpi.h"
#endif /*HAVE_MPI */

#include "moesi.h"

typedef struct dague_ddesc {
    uint32_t myrank;  /**< process rank */
    uint32_t cores;   /**< number of cores used for computation per node */
    uint32_t nodes;   /**< number of nodes involved in the computation */
    uint32_t (*rank_of)(struct dague_ddesc *mat, ...);   /* return the rank of the process owning the data  */
    void *   (*data_of)(struct dague_ddesc *mat, ...);   /* return the pointer to the data possessed locally */
    int32_t  (*vpid_of)(struct dague_ddesc *mat, ...);   /* return the virtual process ID of data possessed locally */
    moesi_map_t* moesi_map; /* the map that tracks accesses to replicates of memory blocks */
#ifdef DAGUE_PROF_TRACE
    uint32_t (*data_key)(struct dague_ddesc *mat, ...); /* return a unique key (unique only for the specified dague_ddesc) associated to a data */
    int (*key_to_string)(struct dague_ddesc *mat, uint32_t datakey, char * buffer, uint32_t buffer_size); /* compute a string in 'buffer' meaningful for profiling about data, return the size of the string */
    char      *key_dim;
    char      *key;
#endif /* DAGUE_PROF_TRACE */
} dague_ddesc_t;

static inline void dague_ddesc_destroy(dague_ddesc_t *d)
{
#if defined(DAGUE_PROF_TRACE)
    if( NULL != d->key_dim )
        free(d->key_dim);
    d->key_dim = NULL;

    if( NULL != d->key )
        free(d->key);
    d->key = NULL;
#else
    (void)d;
#endif
}

#if defined(DAGUE_PROF_TRACE)
#define dague_ddesc_set_key(d, k) do {                                  \
        char dim[strlen(k) + strlen( (d)->key_dim ) + 4];               \
        (d)->key = strdup(k);                                           \
        sprintf(dim, "%s%s", k, (d)->key_dim);                          \
        dague_profiling_add_information( "DIMENSION", dim );            \
    } while(0)
#else
#define dague_ddesc_set_key(d, k) do {} while(0)
#endif

#endif /* _DATA_DISTRIBUTION_H_ */

