/*
 * Copyright (c) 2013-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <dague_config.h>
#include "lifo.h"

#if !defined(DAGUE_ATOMIC_HAS_ATOMIC_CAS_128B)
#warning "DAGuE LIFO is using the keeper_of_the_seven_keys probabilistic atomics, due to lack of CAS 128 bit support"
#endif

static inline void dague_lifo_construct( dague_lifo_t* lifo )
{
    /* Don't allow strange alignemnts */
    if( (0 == lifo->alignment) || (lifo->alignment > 16) ) {
        lifo->alignment = DAGUE_LIFO_ALIGNMENT_DEFAULT;
    }
    DAGUE_LIFO_ITEM_ALLOC( lifo, lifo->lifo_ghost, sizeof(dague_list_item_t) );
    DAGUE_ITEM_ATTACH(lifo, lifo->lifo_ghost);
    lifo->lifo_head = DAGUE_LIFO_HKEY(lifo, lifo->lifo_ghost, 0);
}

static inline void dague_lifo_destruct( dague_lifo_t *lifo )
{
    if( NULL != lifo->lifo_ghost ) {
        DAGUE_ITEM_DETACH(lifo->lifo_ghost);
        DAGUE_LIFO_ITEM_FREE(lifo->lifo_ghost);
    }
}

OBJ_CLASS_INSTANCE(dague_lifo_t, dague_object_t,
                   dague_lifo_construct, dague_lifo_destruct);
