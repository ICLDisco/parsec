/*
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <dague_config.h>
#include "list.h"

/**
 * The list_item object instance.
 */
static inline void
dague_list_item_construct( dague_list_item_t* item )
{
    item->list_prev = item;
    item->list_next = item;
    item->keeper_of_the_seven_keys = 0;
#if defined(DAGUE_DEBUG_ENABLE)
    item->refcount = 0;
    item->belong_to = (void*)0xdeadbeef;
#endif
}

OBJ_CLASS_INSTANCE(dague_list_item_t, dague_object_t, 
                   dague_list_item_construct, NULL);

/**
 * And now the list instance.
 */

static inline void
dague_list_construct( dague_list_t* list )
{
    dague_list_item_construct(&list->ghost_element);
    DAGUE_ITEM_ATTACH(list, &list->ghost_element);
    list->ghost_element.list_next = &list->ghost_element;
    list->ghost_element.list_prev = &list->ghost_element;
    list->atomic_lock = 0;
}

static inline void
dague_list_destruct( dague_list_t* list )
{
    assert(dague_list_is_empty(list));
}

OBJ_CLASS_INSTANCE(dague_list_t, dague_object_t, 
                   dague_list_construct, dague_list_destruct);


