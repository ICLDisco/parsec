/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _DATA_DISTRIBUTION_H_
#define _DATA_DISTRIBUTION_H_

#include "dague_config.h"
#include "dague/types.h"
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

struct dague_device_s;

typedef struct dague_ddesc_s dague_ddesc_t;
typedef int (*dague_memory_region_management_f)(dague_ddesc_t*, struct dague_device_s*);

struct dague_ddesc_s {
    uint32_t            myrank;    /**< process rank */
    uint32_t            cores;     /**< number of cores used for computation per node */
    uint32_t            nodes;     /**< number of nodes involved in the computation */

    /* return a unique key (unique only for the specified dague_ddesc) associated to a data */
    dague_data_key_t (*data_key)(dague_ddesc_t *mat, ...);

    /* return the rank of the process owning the data  */
    uint32_t (*rank_of)(dague_ddesc_t *mat, ...);
    uint32_t (*rank_of_key)(dague_ddesc_t *mat, dague_data_key_t key);

    /* return the pointer to the data possessed locally */
    dague_data_t* (*data_of)(dague_ddesc_t *mat, ...);
    dague_data_t* (*data_of_key)(dague_ddesc_t *mat, dague_data_key_t key);

    /* return the virtual process ID of data possessed locally */
    int32_t  (*vpid_of)(dague_ddesc_t *mat, ...);
    int32_t  (*vpid_of_key)(dague_ddesc_t *mat, dague_data_key_t key);

    /* compute a string in 'buffer' meaningful for profiling about data, return the size of the string */
    int (*key_to_string)(dague_ddesc_t *mat, dague_data_key_t key, char * buffer, uint32_t buffer_size);

    /* Memory management function. They are used to register/unregister the data description
     * with the active devices.
     */
    dague_memory_region_management_f register_memory;
    dague_memory_region_management_f unregister_memory;

    char      *key_base;
#ifdef DAGUE_PROF_TRACE
    char      *key_dim;  /* TODO: Do we really need this field */
#endif /* DAGUE_PROF_TRACE */
};

/**
 * Define a static data representation where all data are local
 * on the current dague_context_t.
 */
extern const dague_ddesc_t dague_static_local_data_ddesc;

static inline void dague_ddesc_destroy(dague_ddesc_t *d)
{
#if defined(DAGUE_PROF_TRACE)
    if( NULL != d->key_dim ) free(d->key_dim);
    d->key_dim = NULL;
#endif
    if( NULL != d->key_base ) free(d->key_base);
    d->key_base = NULL;
}

#if defined(DAGUE_PROF_TRACE)
/* TODO: Fix me pleaseeeeeee */
#define dague_ddesc_set_key(d, k) do {                                  \
        char dim[strlen(k) + strlen( (d)->key_dim ) + 4];               \
        (d)->key_base = strdup(k);                                      \
        sprintf(dim, "%s%s", k, (d)->key_dim);                          \
        dague_profiling_add_information( "DIMENSION", dim );            \
    } while(0)
#else
#define dague_ddesc_set_key(d, k) do {} while(0)
#endif

#endif /* _DATA_DISTRIBUTION_H_ */

