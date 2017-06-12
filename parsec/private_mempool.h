/*
 * Copyright (c) 2009-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef MEMORY_POOL_H_HAS_BEEN_INCLUDED
#define MEMORY_POOL_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"

/** 
 *  @addtogroup parsec_internal_mempool
 *  @{
 */

#include "parsec/class/lifo.h"
#include <stdlib.h>

BEGIN_C_DECLS

typedef struct parsec_memory_pool_t {
    parsec_lifo_t lifo;
    size_t elem_size;
} parsec_memory_pool_t;

extern int
parsec_private_memory_init( parsec_memory_pool_t* pool,
                           size_t size );

static inline void*
parsec_private_memory_pop(parsec_memory_pool_t* pool)
{
    parsec_list_item_t* elem = parsec_lifo_pop(&(pool->lifo));
    if( NULL == elem ) {
        PARSEC_LIFO_ITEM_ALLOC(&(pool->lifo), elem, pool->elem_size );
    }
    return (void*)((char*)elem+sizeof(parsec_list_item_t));
}

static inline void
parsec_private_memory_push(parsec_memory_pool_t* pool, void* memory)
{
    parsec_list_item_t* item = (parsec_list_item_t*)(((intptr_t)memory) - sizeof(parsec_list_item_t));
    parsec_lifo_push( &(pool->lifo), item );
}

extern int parsec_private_memory_fini(parsec_memory_pool_t* pool);

END_C_DECLS

/** @} */

#endif  /* MEMORY_POOL_H_HAS_BEEN_INCLUDED */

