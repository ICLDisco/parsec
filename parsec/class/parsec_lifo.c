/*
 * Copyright (c) 2013-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/class/lifo.h"
#include "parsec/sys/atomic.h"

static inline void parsec_lifo_construct( parsec_lifo_t* lifo )
{
    /* Don't allow strange alignemnts */
    lifo->alignment = PARSEC_LIFO_ALIGNMENT_DEFAULT;
    PARSEC_LIFO_ITEM_ALLOC( lifo, lifo->lifo_ghost, sizeof(parsec_list_item_t) );
    PARSEC_ITEM_ATTACH(lifo, lifo->lifo_ghost);
    lifo->lifo_head.data.item = lifo->lifo_ghost;
    lifo->lifo_head.data.guard.counter = 0;
    /* We cannot use PARSEC_ATOMIC_UNLOCKED for not static initializers
     * so instead we need to clear the state of the lock.
     */
    parsec_atomic_unlock(&lifo->lifo_head.data.guard.lock);
}

static inline void parsec_lifo_destruct( parsec_lifo_t *lifo )
{
    if( NULL != lifo->lifo_ghost ) {
        PARSEC_ITEM_DETACH(lifo->lifo_ghost);
        PARSEC_LIFO_ITEM_FREE(lifo->lifo_ghost);
    }
}

OBJ_CLASS_INSTANCE(parsec_lifo_t, parsec_object_t,
                   parsec_lifo_construct, parsec_lifo_destruct);

