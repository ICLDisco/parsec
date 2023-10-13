/*
 * Copyright (c) 2013-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/runtime.h"
#include "branching_data.h"
#include <stdarg.h>
#include "parsec/data_distribution.h"
#include "parsec/data_internal.h"
#include "parsec/utils/debug.h"

typedef struct {
    parsec_data_collection_t  super;
    parsec_data_t           **data;
    int      nt;
    size_t   size;
    int32_t* ptr;
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

static int32_t vpid_of(parsec_data_collection_t *desc, ...)
{
    int k;
    va_list ap;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    (void)k;

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

    (void)k;
    assert(k >= 0);
    assert(k < dat->nt);
    return parsec_data_create(&dat->data[k], desc, k, &dat->ptr[k*dat->size*sizeof(int32_t)], dat->size * sizeof(int32_t), 0);
}

#if defined(PARSEC_PROF_TRACE)
static parsec_data_key_t data_key(parsec_data_collection_t *desc, ...)
{
    int k;
    va_list ap;

    va_start(ap, desc);
    k = va_arg(ap, int);
    va_end(ap);

    return (uint32_t)k;
}
#endif

parsec_data_collection_t *create_and_distribute_data(int rank, int world, int size, int nb)
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
      d->key_base = NULL;
      d->data_key = data_key;
    }
#endif
    parsec_type_create_contiguous(size, parsec_datatype_int32_t, &d->default_dtt);

    m->data = calloc(sizeof(parsec_data_t*), nb);
    m->nt   = nb;
    m->size = size;
    m->ptr  = (int32_t*)malloc(nb * size * sizeof(int32_t));

    return d;
}

void free_data(parsec_data_collection_t *d)
{
    my_datatype_t *m = (my_datatype_t*)d;
    free(m->ptr);
    for(int i = 0; i < m->nt; i++)
        if(NULL != m->data[i]) parsec_data_destroy(m->data[i]);
    free(m->data);
    parsec_data_collection_destroy(d);
    free(d);
}
