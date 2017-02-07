/*
 * Copyright (c) 2014-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "rtt_data.h"
#include <stdarg.h>
#include "parsec/data_distribution.h"
#include "parsec/data_internal.h"
#include "parsec/debug.h"

#include <assert.h>

typedef struct {
    parsec_ddesc_t        super;
    size_t                size;
    struct parsec_data_s *data;
    uint8_t              *ptr;
} my_datatype_t;

static uint32_t rank_of(parsec_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( (unsigned int)k < dat->super.nodes && k >= 0 );
    (void)dat;
    return k;
}

static uint32_t rank_of_key(parsec_ddesc_t *desc, parsec_data_key_t key)
{
    (void)desc;
    return (uint32_t)key;
}

static int32_t vpid_of(parsec_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( (unsigned int)k < dat->super.nodes && k >= 0 );
    (void)dat; (void)k;
    return 0;
}

static int32_t vpid_of_key(parsec_ddesc_t *desc, parsec_data_key_t key)
{
    (void)desc; (void)key;
    return 0;
}

static parsec_data_t* data_of(parsec_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( (unsigned int)k < dat->super.nodes && k >= 0 );
    return parsec_data_create( &dat->data, desc, k, dat->ptr, dat->size );
}

static parsec_data_t* data_of_key(parsec_ddesc_t *desc, parsec_data_key_t key)
{
    my_datatype_t *dat = (my_datatype_t*)desc;
    return parsec_data_create( &dat->data, desc, key, dat->ptr, dat->size );
}

static uint32_t data_key(parsec_ddesc_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( (unsigned int)k < dat->super.nodes && k >= 0 );
    (void)dat;

    return (uint32_t)k;
}

parsec_ddesc_t *create_and_distribute_data(int rank, int world, int size)
{
    my_datatype_t *m = (my_datatype_t*)calloc(1, sizeof(my_datatype_t));
    parsec_ddesc_t *d = &(m->super);

    d->myrank       = rank;
    d->nodes        = world;
    d->rank_of      = rank_of;
    d->rank_of_key  = rank_of_key;
    d->data_of      = data_of;
    d->data_of_key  = data_of_key;
    d->vpid_of      = vpid_of;
    d->vpid_of_key  = vpid_of_key;
    d->data_key     = data_key;
#if defined(PARSEC_PROF_TRACE)
    asprintf(&d->key_dim, "(%d)", world);
#endif

    m->size = size;
    m->data = NULL;
    m->ptr  = (uint8_t*)malloc(size);

    return d;
}

void free_data(parsec_ddesc_t *d)
{
    my_datatype_t *m = (my_datatype_t*)d;
    if(NULL != m->data) {
        parsec_data_destroy( m->data );
    }
    free(m->ptr);
    m->ptr = NULL;
    parsec_ddesc_destroy(d);
    free(d);
}
