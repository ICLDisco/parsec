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
int dague_freelist_init( dague_freelist_t* freelist, size_t elem_size, int ondemand )
{
    dague_atomic_lifo_construct(&(freelist->lifo));
    freelist->elem_size = elem_size;
    freelist->ondemand = ondemand;
    return 0;
}

/**
 *
 */
int dague_freelist_fini( dague_freelist_t* freelist )
{
    dague_freelist_item_t* item;
    while( NULL != (item = (dague_freelist_item_t*)dague_atomic_lifo_pop(&(freelist->lifo))) ) {
        free(item);
    }
    freelist->elem_size = 0;
    return 0;
}

/**
 *
 */
dague_list_item_t* dague_freelist_get(dague_freelist_t* freelist)
{
    dague_freelist_item_t* item = (dague_freelist_item_t*)dague_atomic_lifo_pop(&(freelist->lifo));

    if( NULL != item ) {
        item->upstream.origin = freelist;
        return (dague_list_item_t*)item;
    }
    if( freelist->ondemand ) {
        item = calloc(1, freelist->elem_size);
        item->upstream.origin = freelist;
    }
    return (dague_list_item_t*)item;
}

/**
 *
 */
int dague_freelist_release( dague_freelist_item_t* item )
{
    dague_atomic_lifo_t* lifo = &(item->upstream.origin->lifo);
    item->upstream.item.list_prev = (dague_list_item_t*)item;
    item->upstream.item.list_next = (dague_list_item_t*)item;
    dague_atomic_lifo_push( lifo, (dague_list_item_t*)item );
    return 0;
}
