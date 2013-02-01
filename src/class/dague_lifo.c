/*
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <dague_config.h>
#include "lifo.h"

#ifdef DAGUE_DEBUG_LIFO_USE_ATOMICS

static inline void dague_lifo_construct( dague_lifo_t* lifo )
{
    OBJ_CONSTRUCT(&lifo->lifo_ghost, dague_list_item_t);
    DAGUE_ITEM_ATTACH(lifo, &lifo->lifo_ghost);
    lifo->lifo_head = &lifo->lifo_ghost;
}

static inline void dague_lifo_destruct( dague_lifo_t *lifo )
{
    DAGUE_ITEM_DETACH(&lifo->lifo_ghost);
}

OBJ_CLASS_INSTANCE(dague_lifo_t, dague_object_t, 
                   dague_lifo_construct, dague_lifo_destruct);

#else /* DAGUE_DEBUG_LIFO_USE_ATOMICS */

OBJ_CLASS_INSTANCE(dague_lifo_t, dague_list_t, 
                   NULL, NULL);

#endif /* DAGUE_DEBUG_LIFO_USE_ATOMICS */


