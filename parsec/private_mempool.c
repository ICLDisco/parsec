/*
 * Copyright (c) 2009-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/private_mempool.h"
#include "parsec/class/lifo.h"
#include "parsec/constants.h"

int
parsec_private_memory_init( parsec_memory_pool_t* pool,
                           size_t size )
{
    PARSEC_OBJ_CONSTRUCT( &(pool->lifo), parsec_lifo_t );
    pool->lifo.alignment = 0;  /* reset the default alignment to the minimum possible */
    pool->elem_size = size + sizeof(parsec_list_item_t);
    return PARSEC_SUCCESS;
}

int parsec_private_memory_fini( parsec_memory_pool_t* pool )
{
    parsec_list_item_t* elem;

    while( NULL != (elem = parsec_lifo_pop(&(pool->lifo))) ) {
        PARSEC_OBJ_RELEASE(elem);
    }
    PARSEC_OBJ_DESTRUCT( &(pool->lifo) );
    return PARSEC_SUCCESS;
}
