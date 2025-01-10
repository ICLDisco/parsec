/*
 * Copyright (c) 2009-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "ctlgat_data.h"
#include "stdarg.h"
#include "parsec/data_distribution.h"
#include "parsec/data_internal.h"
#include "parsec/utils/debug.h"

#include <string.h>
#include <assert.h>
#include <string.h>

typedef struct {
    parsec_data_collection_t super;
    int   seg;
    int   size;
    struct parsec_data_copy_s* data;
    uint32_t* ptr;
} my_datatype_t;

static uint32_t rank_of(parsec_data_collection_t *desc, ...)
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

static int32_t vpid_of(parsec_data_collection_t *desc, ...)
{
    int k;
    va_list ap;
    my_datatype_t *dat = (my_datatype_t*)desc;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    assert( k < dat->size && k >= 0 );
    (void)k; (void)dat;
    return 0;
}

static parsec_data_t* data_of(parsec_data_collection_t *desc, ...)
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
        dat->data = parsec_data_copy_new(NULL, 0, desc->default_dtt, PARSEC_DATA_FLAG_PARSEC_MANAGED);
        dat->data->device_private = dat->ptr;
    }
    return (void*)(dat->data);
}

#if defined(PARSEC_PROF_TRACE)
static parsec_data_key_t data_key(parsec_data_collection_t *desc, ...)
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

parsec_data_collection_t *create_and_distribute_data(int rank, int world, int size, int seg)
{
    my_datatype_t *m = (my_datatype_t*)calloc(1, sizeof(my_datatype_t));
    parsec_data_collection_t *d = &(m->super);
    
    d->myrank = rank;
    d->nodes  = world;
    d->rank_of = rank_of;
    d->data_of = data_of;
    d->vpid_of = vpid_of;
#if defined(PARSEC_PROF_TRACE)
    {
      int len = asprintf(&d->key_dim, "(%d)", size);
      if( -1 == len )
	d->key_dim = NULL;
      d->key_base = strdup("A");
      d->data_key = data_key;
    }
#endif
    parsec_type_create_contiguous(size, parsec_datatype_uint32_t, &d->default_dtt);

    m->size = size;
    m->seg  = seg;
    m->data = NULL;
    m->ptr = (uint32_t*)calloc(seg * size, sizeof(uint32_t) );

    return d;
}

void free_data(parsec_data_collection_t *d)
{
    my_datatype_t *m = (my_datatype_t*)d;
    if(NULL != m->data) {
        PARSEC_DATA_COPY_RELEASE(m->data);
    }
    free(m->ptr);
    parsec_data_collection_destroy(d);
    free(d);
}
