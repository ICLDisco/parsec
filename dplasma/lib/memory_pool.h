/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef MEMORY_POOL_H_HAS_BEEN_INCLUDED
#define MEMORY_POOL_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "lifo.h"
#include <stdlib.h>

typedef struct dague_memory_pool_t {
    dague_atomic_lifo_t lifo;
    size_t elem_size;
} dague_memory_pool_t;

extern int
dague_private_memory_init( dague_memory_pool_t* pool,
                           size_t size );

static inline void*
dague_private_memory_pop(dague_memory_pool_t* pool)
{
    dague_list_item_t* elem = dague_atomic_lifo_pop(&(pool->lifo));
    if( NULL == elem ) {
        DAGUE_LIFO_ELT_ALLOC(elem, pool->elem_size );
    }
    return (void*)((char*)elem+sizeof(dague_list_item_t));
}

static inline void
dague_private_memory_push(dague_memory_pool_t* pool, void* memory)
{
    dague_list_item_t* item = DAGUE_LIST_ITEM_SINGLETON( (((char*)memory)-sizeof(dague_list_item_t)) );
    dague_atomic_lifo_push( &(pool->lifo), item );
}

extern int dague_private_memory_fini(dague_memory_pool_t* pool);

#endif  /* MEMORY_POOL_H_HAS_BEEN_INCLUDED */

