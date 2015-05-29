/*
 * Copyright (c) 2014-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "rtt_data.h"
#include <stdarg.h>
#include "dague/data_distribution.h"
#include "dague/data_internal.h"
#include "dague/debug.h"

#include <assert.h>

typedef struct {
    dague_ddesc_t super;
    int   seg;
    int   size;
    struct dague_data_s     * data;
    struct dague_data_copy_s* data_copy;
    uint32_t* ptr;
} my_datatype_t;

static uint32_t rank_of(dague_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( k < dat->size && k >= 0 );
    (void)dat;
    return k;
}

static int32_t vpid_of(dague_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( k < dat->size && k >= 0 );
    (void)dat; (void)k;
    return 0;
}

static dague_data_t* data_of(dague_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( k < dat->size && k >= 0 );
    (void)k;
    if(NULL == dat->data) {
        dat->data = dague_data_new();
        dat->data_copy = dague_data_copy_new(dat->data, 0);
        dat->data_copy->device_private = dat->ptr;
    }
    return dat->data;
}

#if defined(DAGUE_PROF_TRACE)
static uint32_t data_key(dague_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( k < dat->size && k >= 0 ); (void)dat;

    return (uint32_t)k;
}
#endif

dague_ddesc_t *create_and_distribute_data(int rank, int world, int size, int seg)
{
    my_datatype_t *m = (my_datatype_t*)calloc(1, sizeof(my_datatype_t));
    dague_ddesc_t *d = &(m->super);

    d->myrank = rank;
    d->nodes  = world;
    d->rank_of = rank_of;
    d->data_of = data_of;
    d->vpid_of = vpid_of;
#if defined(DAGUE_PROF_TRACE)
    asprintf(&d->key_dim, "(%d)", size);
    d->key_base = strdup("A");
    d->data_key = data_key;
#endif

    m->size = size;
    m->seg  = seg;
    m->data      = NULL;
    m->data_copy = NULL;
    m->ptr = (uint32_t*)calloc(seg * size, sizeof(uint32_t) );

    return d;
}

void free_data(dague_ddesc_t *d)
{
    my_datatype_t *m = (my_datatype_t*)d;
    if(NULL != m->data_copy) {
        dague_data_copy_detach(m->data, m->data_copy, 0);
        DAGUE_DATA_COPY_RELEASE(m->data_copy);
        OBJ_RELEASE(m->data);
        m->data_copy = NULL;
        m->data = NULL;
    }
    free(m->ptr);
    m->ptr = NULL;
    dague_ddesc_destroy(d);
    free(d);
}
