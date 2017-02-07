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
    parsec_ddesc_t super;
    size_t size;
    struct parsec_data_s     * data;
    struct parsec_data_copy_s* data_copy;
    uint32_t* ptr;
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

#if defined(PARSEC_PROF_TRACE)
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
#endif

parsec_ddesc_t *create_and_distribute_data(int rank, int world, int size)
{
    my_datatype_t *m = (my_datatype_t*)calloc(1, sizeof(my_datatype_t));
    parsec_ddesc_t *d = &(m->super);

    d->myrank = rank;
    d->nodes  = world;
    d->rank_of = rank_of;
    d->data_of = data_of;
    d->vpid_of = vpid_of;
#if defined(PARSEC_PROF_TRACE)
    asprintf(&d->key_dim, "(%d)", world);
    d->key_base = strdup("A");
    d->data_key = data_key;
#endif

    m->size = size;
    m->data      = NULL;
    m->data_copy = NULL;
    m->ptr = (uint32_t*)calloc(size, 1);

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
