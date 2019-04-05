/*
 * Copyright (c) 2010-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#ifndef _PARSEC_DATA_DISTRIBUTION_H_
#define _PARSEC_DATA_DISTRIBUTION_H_

#include "parsec/data.h"
#include "parsec/class/parsec_hash_table.h"

BEGIN_C_DECLS

struct parsec_device_module_s;
typedef int (*parsec_memory_region_management_f)(parsec_data_collection_t*, struct parsec_device_module_s*);

typedef uint8_t parsec_memory_registration_status_t;
#define    PARSEC_MEMORY_STATUS_UNREGISTERED      ((parsec_memory_registration_status_t)0x0)
#define    PARSEC_MEMORY_STATUS_REGISTERED        ((parsec_memory_registration_status_t)0x1)

typedef uint64_t parsec_dc_key_t;

struct parsec_data_collection_s {
    parsec_dc_key_t     dc_id;     /**< DC are uniquely globally consistently named */
    parsec_hash_table_item_t ht_item; /**< to be pushable in a hash table */

    uint32_t            myrank;    /**< process rank */
    uint32_t            nodes;     /**< number of nodes involved in the computation */

    /* This hash table book keep dtd interface */
    parsec_hash_table_t *tile_h_table;

    /* return a unique key (unique only for the specified parsec_dc) associated to a data */
    parsec_data_key_t (*data_key)(parsec_data_collection_t *d, ...);

    /* return the rank of the process owning the data  */
    uint32_t (*rank_of)(parsec_data_collection_t *d, ...);
    uint32_t (*rank_of_key)(parsec_data_collection_t *d, parsec_data_key_t key);

    /* return the pointer to the data possessed locally */
    parsec_data_t* (*data_of)(parsec_data_collection_t *d, ...);
    parsec_data_t* (*data_of_key)(parsec_data_collection_t *d, parsec_data_key_t key);

    /* return the virtual process ID of data possessed locally */
    int32_t  (*vpid_of)(parsec_data_collection_t *d, ...);
    int32_t  (*vpid_of_key)(parsec_data_collection_t *d, parsec_data_key_t key);

    /* Memory management function. They are used to register/unregister the data description
     * with the active devices.
     */
    parsec_memory_region_management_f register_memory;
    parsec_memory_region_management_f unregister_memory;
    parsec_memory_registration_status_t memory_registration_status;

    char      *key_base;

    /* compute a string in 'buffer' meaningful for profiling about data, return the size of the string */
    int (*key_to_string)(parsec_data_collection_t *d, parsec_data_key_t key, char * buffer, uint32_t buffer_size);
    char      *key_dim;
    char      *key;
};

/**
 * Initialize/destroy the parsec_dc to default values.
 */
void
parsec_data_collection_init(parsec_data_collection_t *d,
                            int nodes, int myrank );
void
parsec_data_collection_destroy(parsec_data_collection_t *d);

PARSEC_DECLSPEC int
parsec_data_dist_init(void);

PARSEC_DECLSPEC int
parsec_data_dist_fini(void);

int
parsec_dc_register_id(parsec_dc_t* dc, parsec_dc_key_t key);

int
parsec_dc_unregister_id(parsec_dc_key_t key);

parsec_dc_t *
parsec_dc_lookup(parsec_dc_key_t key);

#if defined(PARSEC_PROF_TRACE)
void parsec_data_collection_set_key( parsec_data_collection_t* d, char* name);
#else
#define parsec_data_collection_set_key(d, k) do {} while(0)
#endif

END_C_DECLS

#endif /* _DATA_DISTRIBUTION_H_ */

