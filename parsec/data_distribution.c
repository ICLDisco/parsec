/*
 * Copyright (c) 2017-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/data_distribution.h"
#include "parsec/utils/debug.h"

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

/* To generate consistent global id for each data collection (DC) */
static parsec_hash_table_t *parsec_dc_hash_table = NULL;
static int parsec_dc_hash_table_size = 101;

static parsec_key_fn_t dc_key_fns = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_print = parsec_hash_table_generic_64bits_key_print,
    .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

int
parsec_data_dist_init(void)
{
    int count;
    parsec_dc_hash_table = PARSEC_OBJ_NEW(parsec_hash_table_t);
    for(count = 1; count < 16 && (1<<count)<parsec_dc_hash_table_size; count++) /* nothing */;
    parsec_hash_table_init(parsec_dc_hash_table,
                           offsetof(parsec_dc_t, ht_item),
                           count,
                           dc_key_fns,
                           parsec_dc_hash_table);
    return 1;
}

int
parsec_data_dist_fini(void)
{
    parsec_hash_table_fini(parsec_dc_hash_table);
    return 1;
}

int
parsec_dc_register_id(parsec_dc_t* dc, parsec_dc_key_t key)
{
    if( NULL == parsec_dc_hash_table ) {
        /* Do not call this function before initializing the runtime */
        return PARSEC_ERROR;
    }
#if defined(PARSEC_DEBUG_PARANOID)
    parsec_dc_t* registered_dc = (parsec_dc_t*)parsec_hash_table_find(parsec_dc_hash_table, key);
    if( dc == registered_dc ) {
        /* complain but allow the code to continue */
        parsec_inform("Re-registering a data collection (%p) with the same key (%lld). This operation "
                      "is unnecessary and can lead to unexpected consequences. Please correct "
                      "the code", dc, (long long)key);
    } else if( NULL != registered_dc ) {
        parsec_warning("Registering a data collection with an already existing key ID (%lld). "
                       "The ID keys must be unique to avoid collisions between different "
                       "data collections.", (long long)key);
        return PARSEC_ERROR;
    }
#endif  /* defined(PARSEC_DEBUG_PARANOID) */
    dc->ht_item.key = key;
    dc->dc_id = key;
    /* We do not have any function to check atomically if there is a collision
     * so for now we replace the old entry if there is request for
     * a new registration with same id.
     */
    parsec_hash_table_insert(parsec_dc_hash_table, &dc->ht_item);
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "register dc_t %p with key %lld", dc, (long long)key);
    return 1;
}

int
parsec_dc_unregister_id(parsec_dc_key_t key)
{
    if( NULL == parsec_dc_hash_table ) {
        /* Do not call this function before initializing the runtime */
        return PARSEC_ERROR;
    }
    parsec_dc_t* registered_dc = parsec_hash_table_remove(parsec_dc_hash_table, key);
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "unregister dc_t %p with key %lld", registered_dc, (long long)key);
    return (NULL != registered_dc ? PARSEC_SUCCESS : PARSEC_ERROR);
}

/* Retrieve the local DC attached to a unique dc id */
parsec_dc_t *
parsec_dc_lookup(parsec_dc_key_t key)
{
    return (parsec_dc_t *)parsec_hash_table_nolock_find( parsec_dc_hash_table, key );
}

void
parsec_data_collection_init(parsec_data_collection_t *d,
                            int nodes, int myrank )
{
    memset( d, 0, sizeof(parsec_data_collection_t) );

    d->nodes  = nodes;
    d->myrank = myrank;
    d->tile_h_table = NULL;
    d->memory_registration_status = PARSEC_MEMORY_STATUS_UNREGISTERED;
    d->default_dtt = PARSEC_DATATYPE_NULL;
}

void
parsec_data_collection_destroy(parsec_data_collection_t *d)
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

void parsec_data_collection_set_key( parsec_data_collection_t* d, const char* name)
{
    char *kdim = (NULL != d->key_dim)? d->key_dim: "";
    char dim[strlen(name) + strlen(kdim) + 4];
    (d)->key_base = strdup(name);
    sprintf(dim, "%s%s", name, kdim);
    parsec_profiling_add_information( "DIMENSION", dim );
}
#endif  /* defined(PARSEC_PROF_TRACE) */
