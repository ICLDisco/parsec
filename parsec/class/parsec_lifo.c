/*
 * Copyright (c) 2013-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
/**
 * Define LIFO_STATIC_INLINE to empty in order to generate the fallback implementation for
 * the atomic operations. We need to have this version to be able to compile and link
 * with a different compiler than the one PaRSEC has been compiled with. This approach also
 * covers the case of including PaRSEC header files from C++ application.
 *
 * Note that we do not change ATOMIC_STATIC_INLINE because we do want to use the
 * fast atomics in this file, the LIFO interface itself will be produced as functions
 * instead of individual atomic operations.
 */
#define LIFO_STATIC_INLINE
#include "parsec/sys/atomic.h"
#include "parsec/class/lifo.h"

static inline void parsec_lifo_construct( parsec_lifo_t* lifo )
{
    /* Don't allow strange alignemnts */
    lifo->alignment = PARSEC_LIFO_ALIGNMENT_DEFAULT;
    lifo->lifo_ghost = parsec_lifo_item_alloc( lifo, sizeof(parsec_list_item_t) );
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
        parsec_lifo_item_free(lifo->lifo_ghost);
    }
}

PARSEC_OBJ_CLASS_INSTANCE(parsec_lifo_t, parsec_object_t,
                   parsec_lifo_construct, parsec_lifo_destruct);

