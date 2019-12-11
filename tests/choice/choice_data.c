/*
 * Copyright (c) 2009-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "choice_data.h"
#include "parsec/data_distribution.h"
#include "parsec/data_internal.h"
#include <stdarg.h>
#include <stdio.h>
#include <assert.h>

typedef struct {
    parsec_data_collection_t   super;
    uint32_t        size;
    parsec_data_t  **data_map;
    int32_t        *data;
} my_datatype_t;

static uint32_t rank_of(parsec_data_collection_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    return k % dat->super.nodes;
}

static inline parsec_data_t*
get_or_create_data(my_datatype_t* dat, uint32_t pos)
{
    parsec_data_t* data = dat->data_map[pos];
    assert(pos <= dat->size);

    if( NULL == data ) {
        parsec_data_copy_t* data_copy = PARSEC_OBJ_NEW(parsec_data_copy_t);
        data = PARSEC_OBJ_NEW(parsec_data_t);

        data_copy->coherency_state = PARSEC_DATA_COHERENCY_OWNED;
        data_copy->original = data;
        data_copy->device_private = &dat->data[pos];

        data->owner_device = 0;
        data->key = pos;
        data->nb_elts = 1;
        data->device_copies[0] = data_copy;

        if( !parsec_atomic_cas_ptr(&dat->data_map[pos], NULL, data) ) {
            free(data_copy);
            free(data);
            data = dat->data_map[pos];
        }
    } else {
        /* Do we have a copy of this data */
        if( NULL == data->device_copies[0] ) {
            parsec_data_copy_t* data_copy = parsec_data_copy_new(data, 0);
            data_copy->device_private = &dat->data[pos];
        }
    }
    return data;
}

static parsec_data_t* data_of(parsec_data_collection_t *desc, ...)
{
    int k;

    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    (void)k;

    return get_or_create_data(dat, k);
} 

static int vpid_of(parsec_data_collection_t *desc, ...)
{
    int k;

    va_list ap;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    (void)k;

    return 0;
}

#if defined(PARSEC_PROF_TRACE)
static parsec_data_key_t data_key(struct parsec_data_collection_s *desc, ...)
{
    int k;
    va_list ap;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    return (uint32_t)k;
}
#endif

parsec_data_collection_t *create_and_distribute_data(int rank, int world, int size)
{
    my_datatype_t *m = (my_datatype_t*)calloc(1, sizeof(my_datatype_t));
    parsec_data_collection_t *d = &(m->super);

    d->myrank  = rank;
    d->nodes   = world;
    d->rank_of = rank_of;
    d->data_of = data_of;
    d->vpid_of = vpid_of;
#if defined(PARSEC_PROF_TRACE)
    asprintf(&d->key_dim, "(%d)", size);
    d->key = NULL;
    d->data_key = data_key;
#endif

    m->size     = size;
    m->data_map = (parsec_data_t**)calloc(size, sizeof(parsec_data_t*));
    m->data     = (int32_t*)malloc(size * sizeof(int32_t));

    return d;
}

void free_data(parsec_data_collection_t *d)
{
    my_datatype_t *m = (my_datatype_t*)d;
    free(m->data);
    parsec_data_collection_destroy(d);
    free(d);
}
