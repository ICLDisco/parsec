/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_LINKED_LIST_H_HAS_BEEN_INCLUDED
#define DAGUE_LINKED_LIST_H_HAS_BEEN_INCLUDED

#include "atomic.h"
#include "lifo.h"

typedef struct dague_linked_list_t {
    dague_list_item_t  ghost_element;
    uint32_t atomic_lock;
} dague_linked_list_t;

static inline void dague_linked_list_construct( dague_linked_list_t* linked_list )
{
    linked_list->ghost_element.list_next = &(linked_list->ghost_element);
    linked_list->ghost_element.list_prev = &(linked_list->ghost_element);
    linked_list->atomic_lock = 0;
}

static inline void dague_linked_list_item_construct( dague_list_item_t *item )
{
    item->list_prev = item;
    item->list_next = item;
}

static inline int dague_linked_list_is_empty( dague_linked_list_t * linked_list )
{
    return linked_list->ghost_element.list_next != &(linked_list->ghost_element);
}

static inline void 
dague_linked_list_add_head( dague_linked_list_t * linked_list,
                            dague_list_item_t *item )
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    item->list_prev = &(linked_list->ghost_element);
    item->list_next = linked_list->ghost_element.list_next;
    linked_list->ghost_element.list_next->list_prev = item;
    linked_list->ghost_element.list_next = item;
    DAGUE_ATTACH_ELEMS(linked_list, item);
    dague_atomic_unlock(&(linked_list->atomic_lock));
}

static inline void 
dague_linked_list_add_tail( dague_linked_list_t * linked_list,
                            dague_list_item_t *item )
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    item->list_next = &(linked_list->ghost_element);
    item->list_prev = linked_list->ghost_element.list_prev;
    linked_list->ghost_element.list_prev->list_next = item;
    linked_list->ghost_element.list_prev = item;
    DAGUE_ATTACH_ELEMS(linked_list, item);
    dague_atomic_unlock(&(linked_list->atomic_lock));
}

static inline dague_list_item_t*
dague_linked_list_remove_item( dague_linked_list_t * linked_list,
                               dague_list_item_t* item)
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
    item->list_next = item;
    item->list_prev = item;
    dague_atomic_unlock(&(linked_list->atomic_lock));
    DAGUE_DETACH_ELEM(item);
    return item;
}

static inline dague_list_item_t*
dague_linked_list_remove_head( dague_linked_list_t * linked_list )
{
    dague_list_item_t* item;

    dague_atomic_lock(&(linked_list->atomic_lock));
    item = (dague_list_item_t*)linked_list->ghost_element.list_next;
    linked_list->ghost_element.list_next = item->list_next;
    item->list_next->list_prev = &linked_list->ghost_element;
    item->list_prev = item;
    item->list_next = item;
    dague_atomic_unlock(&(linked_list->atomic_lock));
    if( &(linked_list->ghost_element) != item ) {
        DAGUE_DETACH_ELEM(item);
        return item;
    }
    return NULL;
}

static inline dague_list_item_t*
dague_linked_list_remove_tail( dague_linked_list_t * linked_list )
{
    dague_list_item_t* item;

    dague_atomic_lock(&(linked_list->atomic_lock));
    item = (dague_list_item_t*)linked_list->ghost_element.list_prev;
    linked_list->ghost_element.list_prev = item->list_prev;
    item->list_prev->list_next = &linked_list->ghost_element;
    item->list_prev = item;
    item->list_next = item;
    dague_atomic_unlock(&(linked_list->atomic_lock));
    if( &(linked_list->ghost_element) != item ) {
        DAGUE_DETACH_ELEM(item);
        return item;
    }
    return NULL;
}

#endif  /* DAGUE_LINKED_LIST_H_HAS_BEEN_INCLUDED */
