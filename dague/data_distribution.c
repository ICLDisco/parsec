/*
 * Copyright (c) 2010-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2016      Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                         Univ. Bordeaux. All rights reserved.
 */

#include "dague/data.h"
#include "dague/data_distribution.h"

/**
 * Initialize all the fields of the dague_ddesc with the default descriptor.
 */
void
dague_ddesc_init(dague_ddesc_t *d,
                 int nodes,
                 int myrank)
{
    memset( d, 0, sizeof(dague_ddesc_t) );

    d->nodes  = nodes;
    d->myrank = myrank;
    d->memory_registration_status = MEMORY_STATUS_UNREGISTERED;
}

void
dague_ddesc_destroy(dague_ddesc_t *d)
{
#if defined(DAGUE_PROF_TRACE)
    if( NULL != d->key_dim ) free(d->key_dim);
    d->key_dim = NULL;
#endif
    if( NULL != d->key_base ) free(d->key_base);
    d->key_base = NULL;
}

/**
 * Set of function that describes a default descriptor with only one fake data
 * of size 0, that is owned by everyone nodes on the VP 0.
 */
dague_data_t *fake_data = NULL;

static dague_data_key_t
dague_ddesc_default_data_key(dague_ddesc_t *d, ...)
{
    (void)d;
    return 0;
}

static uint32_t
dague_ddesc_default_rank_of_key(dague_ddesc_t *d,
                                dague_data_key_t key)
{
    (void)key;
    return d->myrank;
}

static uint32_t
dague_ddesc_default_rank_of(dague_ddesc_t *d, ...)
{
    return dague_ddesc_default_rank_of_key(d, 0);
}

static dague_data_t *
dague_ddesc_default_data_of_key(dague_ddesc_t *d,
                                dague_data_key_t key)
{
    if (fake_data == NULL) {
        return dague_data_create( &fake_data, d, key,
                                  NULL, 0 );
    }
    return fake_data;
}

static dague_data_t *
dague_ddesc_default_data_of(dague_ddesc_t *d, ...)
{
    return dague_ddesc_default_data_of_key(d, 0);
}

static int32_t
dague_ddesc_default_vpid_of_key(dague_ddesc_t *d,
                                dague_data_key_t key)
{
    (void)d;
    (void)key;
    return 0;
}

static int32_t
dague_ddesc_default_vpid_of(dague_ddesc_t *d, ...)
{
    return dague_ddesc_default_vpid_of_key(d, 0);
}

#if defined(DAGUE_PROF_TRACE)
static int
dague_ddesc_default_key_to_string(struct dague_ddesc_s *desc,
                                  uint32_t datakey,
                                  char * buffer,
                                  uint32_t buffer_size)
{
    int res;
    res = snprintf(buffer, buffer_size, "(%u)", datakey);
    if (res < 0) {
        printf("(dague_ddesc_default_key_to_string) Error with key: %u\n", datakey);
    }
    return res;
}
#endif /* defined(DAGUE_PROF_TRACE) */

/**
 * Initialize all the fields of the dague_ddesc with the default descriptor.
 */
void
dague_ddesc_default_init(dague_ddesc_t *d,
                         int nodes,
                         int myrank)
{
    dague_ddesc_init(d, nodes, myrank);

    d->data_key    = dague_ddesc_default_data_key;
    d->rank_of     = dague_ddesc_default_rank_of;
    d->rank_of_key = dague_ddesc_default_rank_of_key;
    d->data_of     = dague_ddesc_default_data_of;
    d->data_of_key = dague_ddesc_default_data_of_key;
    d->vpid_of     = dague_ddesc_default_vpid_of;
    d->vpid_of_key = dague_ddesc_default_vpid_of_key;

#if defined(DAGUE_PROF_TRACE)
    d->key_to_string = dague_ddesc_default_key_to_string;
#endif
}

void
dague_ddesc_default_destroy(dague_ddesc_t *d)
{
    dague_ddesc_destroy(d);
}

void
dague_ddesc_set_empty_data(dague_ddesc_t *d)
{
    d->data_of     = dague_ddesc_default_data_of;
    d->data_of_key = dague_ddesc_default_data_of_key;
}

void
dague_ddesc_set_local_distribution(dague_ddesc_t *d)
{
    d->rank_of     = dague_ddesc_default_rank_of;
    d->rank_of_key = dague_ddesc_default_rank_of_key;
}

void
dague_ddesc_set_no_vpid(dague_ddesc_t *d)
{
    d->vpid_of     = dague_ddesc_default_vpid_of;
    d->vpid_of_key = dague_ddesc_default_vpid_of_key;
}
