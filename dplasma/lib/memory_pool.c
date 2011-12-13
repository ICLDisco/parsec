/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "memory_pool.h"

int
dague_private_memory_init( dague_memory_pool_t* pool,
                           size_t size )
{
    dague_atomic_lifo_construct( &(pool->lifo) );
    pool->elem_size = size + sizeof(dague_list_item_t);
    return 0;
}

int dague_private_memory_fini( dague_memory_pool_t* pool )
{
    dague_list_item_t* elem;

    while( NULL != (elem = dague_atomic_lifo_pop(&(pool->lifo))) ) {
        DAGUE_LIFO_ELT_FREE(elem);
    }
    dague_atomic_lifo_destruct( &(pool->lifo) );
    return 0;
}
