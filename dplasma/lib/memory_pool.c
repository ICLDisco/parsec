/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "memory_pool.h"

#ifndef max
#define max(__a, __b) ( ( (__a) > (__b) ) ? (__a) : (__b) )
#endif

int
dague_private_memory_init( dague_memory_pool_t* pool,
                           size_t size )
{
    dague_atomic_lifo_construct( &(pool->lifo) );
    pool->elem_size = max( size, sizeof(dague_list_item_t) );
    return 0;
}

int dague_private_memory_fini( dague_memory_pool_t* pool )
{
    dague_list_item_t* elem;

    while( NULL != (elem = dague_atomic_lifo_pop(&(pool->lifo))) ) {
        free(elem);
    }
    return 0;
}
