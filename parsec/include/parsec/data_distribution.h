/*
 * Copyright (c) 2010-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _DATA_DISTRIBUTION_H_
#define _DATA_DISTRIBUTION_H_

#include "parsec_config.h"
#include "parsec/types.h"

#if defined(PARSEC_HAVE_STDARG_H)
#include <stdarg.h>
#endif  /* defined(PARSEC_HAVE_STDARG_H) */
#if defined(PARSEC_HAVE_UNISTD_H)
#include <unistd.h>
#endif  /* defined(PARSEC_HAVE_UNISTD_H) */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct parsec_device_s;
typedef int (*parsec_memory_region_management_f)(parsec_ddesc_t*, struct parsec_device_s*);

typedef uint8_t memory_registration_status_t;
#define    MEMORY_STATUS_UNREGISTERED      ((memory_registration_status_t)0x0)
#define    MEMORY_STATUS_REGISTERED        ((memory_registration_status_t)0x1)

BEGIN_C_DECLS

struct parsec_ddesc_s {
    uint32_t            myrank;    /**< process rank */
    uint32_t            nodes;     /**< number of nodes involved in the computation */

    /* return a unique key (unique only for the specified parsec_ddesc) associated to a data */
    parsec_data_key_t (*data_key)(parsec_ddesc_t *d, ...);

    /* return the rank of the process owning the data  */
    uint32_t (*rank_of)(parsec_ddesc_t *d, ...);
    uint32_t (*rank_of_key)(parsec_ddesc_t *d, parsec_data_key_t key);

    /* return the pointer to the data possessed locally */
    parsec_data_t* (*data_of)(parsec_ddesc_t *d, ...);
    parsec_data_t* (*data_of_key)(parsec_ddesc_t *d, parsec_data_key_t key);

    /* return the virtual process ID of data possessed locally */
    int32_t  (*vpid_of)(parsec_ddesc_t *d, ...);
    int32_t  (*vpid_of_key)(parsec_ddesc_t *d, parsec_data_key_t key);

    /* Memory management function. They are used to register/unregister the data description
     * with the active devices.
     */
    parsec_memory_region_management_f register_memory;
    parsec_memory_region_management_f unregister_memory;
    memory_registration_status_t memory_registration_status;

    char      *key_base;

#ifdef PARSEC_PROF_TRACE
    /* compute a string in 'buffer' meaningful for profiling about data, return the size of the string */
    int (*key_to_string)(parsec_ddesc_t *d, parsec_data_key_t key, char * buffer, uint32_t buffer_size);
    char      *key_dim;
    char      *key;
#endif /* PARSEC_PROF_TRACE */
};

/**
 * Initialize/destroy the parsec_ddesc to default values.
 */
static inline void
parsec_ddesc_init(parsec_ddesc_t *d,
                 int nodes, int myrank )
{
    memset( d, 0, sizeof(parsec_ddesc_t) );

    d->nodes  = nodes;
    d->myrank = myrank;
    d->memory_registration_status = MEMORY_STATUS_UNREGISTERED;
}

static inline void
parsec_ddesc_destroy(parsec_ddesc_t *d)
{
#if defined(PARSEC_PROF_TRACE)
    if( NULL != d->key_dim ) free(d->key_dim);
    d->key_dim = NULL;
#endif
    if( NULL != d->key_base ) free(d->key_base);
    d->key_base = NULL;
}

#if defined(PARSEC_PROF_TRACE)
#include "parsec/profiling.h"
/* TODO: Fix me pleaseeeeeee */
#define parsec_ddesc_set_key(d, k) do {                                  \
        char dim[strlen(k) + strlen( (d)->key_dim ) + 4];               \
        (d)->key_base = strdup(k);                                      \
        sprintf(dim, "%s%s", k, (d)->key_dim);                          \
        parsec_profiling_add_information( "DIMENSION", dim );            \
    } while(0)
#else
#define parsec_ddesc_set_key(d, k) do {} while(0)
#endif

END_C_DECLS

#endif /* _DATA_DISTRIBUTION_H_ */

