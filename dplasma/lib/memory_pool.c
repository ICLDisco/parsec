/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "memory_pool.h"
#include "lifo.h"

int
dague_private_memory_init( dague_memory_pool_t* pool,
                           size_t size )
{
    pool->lifo.alignment = 0;
    OBJ_CONSTRUCT( &(pool->lifo), dague_lifo_t );
    pool->elem_size = size + sizeof(dague_list_item_t);
    return 0;
}

int dague_private_memory_fini( dague_memory_pool_t* pool )
{
    dague_list_item_t* elem;

    while( NULL != (elem = dague_lifo_pop(&(pool->lifo))) ) {
        OBJ_RELEASE(elem);
    }
    OBJ_DESTRUCT( &(pool->lifo) );
    return 0;
}
