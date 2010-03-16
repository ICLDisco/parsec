/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */
#include "freelist.h"
#include <stdlib.h>

/**
 *
 */
int dplasma_freelist_init( dplasma_freelist_t* freelist, size_t elem_size )
{
    dplasma_atomic_lifo_construct(&(freelist->lifo));
    freelist->elem_size = elem_size;
    return 0;
}

/**
 *
 */
int dplasma_freelist_fini( dplasma_freelist_t* freelist )
{
    dplasma_freelist_item_t* item;
    while( NULL != (item = (dplasma_freelist_item_t*)dplasma_atomic_lifo_pop(&(freelist->lifo))) ) {
        free(item);
    }
    freelist->elem_size = 0;
    return 0;
}

/**
 *
 */
dplasma_list_item_t* dplasma_freelist_get(dplasma_freelist_t* freelist)
{
    dplasma_freelist_item_t* item = (dplasma_freelist_item_t*)dplasma_atomic_lifo_pop(&(freelist->lifo));
    if( NULL != item ) {
        item = calloc(1, freelist->elem_size);
    }
    item->upstream.origin = freelist;
    return (dplasma_list_item_t*)item;
}

/**
 *
 */
int dplasma_freelist_release( dplasma_freelist_item_t* item )
{
    dplasma_atomic_lifo_push( &(item->upstream.origin->lifo), (dplasma_list_item_t*)item );
    return 0;
}
