/*                                                                                                                                                                    
 * Copyright (c) 2010      The University of Tennessee and The University                                                                                             
 *                         of Tennessee Research Foundation.  All rights                                                                                              
 *                         reserved.                                                                                                                                  
 */

#ifndef DPLASMA_LINKED_LIST_H_HAS_BEEN_INCLUDED
#define DPLASMA_LINKED_LIST_H_HAS_BEEN_INCLUDED

#include "atomic.h"
#include "lifo.h"

typedef struct dplasma_linked_list_t {
    dplasma_list_item_t  ghost_element;
    uint32_t atomic_lock;
} dplasma_linked_list_t;

static inline void dplasma_linked_list_construct( dplasma_linked_list_t* linked_list )
{
    linked_list->ghost_element.list_next = &(linked_list->ghost_element);
    linked_list->ghost_element.list_prev = &(linked_list->ghost_element);
    linked_list->atomic_lock = 0;
}

static inline void dplamsa_linked_list_item_construct( dplasma_list_item_t *item )
{
    item->list_prev = item;
    item->list_next = item;
}

static inline int dplasma_linked_list_is_empty( dplasma_linked_list_t * linked_list )
{
    return linked_list->ghost_element.list_next != &(linked_list->ghost_element);
}

static inline void 
dplasma_linked_list_add_head( dplasma_linked_list_t * linked_list,
                              dplasma_list_item_t *item )
{
    dplasma_atomic_lock(&(linked_list->atomic_lock));
    item->list_prev = &(linked_list->ghost_element);
    item->list_next = linked_list->ghost_element.list_next;
    linked_list->ghost_element.list_next->list_prev = item;
    linked_list->ghost_element.list_next = item;
    dplasma_atomic_unlock(&(linked_list->atomic_lock));
}

static inline void 
dplasma_linked_list_add_tail( dplasma_linked_list_t * linked_list,
                              dplasma_list_item_t *item )
{
    dplasma_atomic_lock(&(linked_list->atomic_lock));
    item->list_next = &(linked_list->ghost_element);
    item->list_prev = linked_list->ghost_element.list_prev;
    linked_list->ghost_element.list_prev->list_next = item;
    linked_list->ghost_element.list_prev = item;
    dplasma_atomic_unlock(&(linked_list->atomic_lock));
}

static inline dplasma_list_item_t*
dplasma_linked_list_remove_item( dplasma_linked_list_t * linked_list,
                                 dplasma_list_item_t* item)
{
    dplasma_atomic_lock(&(linked_list->atomic_lock));
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
    item->list_next = item;
    item->list_prev = item;
    dplasma_atomic_unlock(&(linked_list->atomic_lock));
    return item;
}

static inline dplasma_list_item_t*
dplasma_linked_list_remove_head( dplasma_linked_list_t * linked_list )
{
    dplasma_list_item_t* item;

    dplasma_atomic_lock(&(linked_list->atomic_lock));
    item = (dplasma_list_item_t*)linked_list->ghost_element.list_next;
    linked_list->ghost_element.list_next = item->list_next;
    item->list_next->list_prev = &linked_list->ghost_element;
    item->list_prev = item;
    item->list_next = item;
    dplasma_atomic_unlock(&(linked_list->atomic_lock));

    return (item == &(linked_list->ghost_element) ? NULL : item);
}

static inline dplasma_list_item_t*
dplasma_linked_list_remove_tail( dplasma_linked_list_t * linked_list )
{
    dplasma_list_item_t* item;

    dplasma_atomic_lock(&(linked_list->atomic_lock));
    item = (dplasma_list_item_t*)linked_list->ghost_element.list_prev;
    linked_list->ghost_element.list_prev = item->list_prev;
    item->list_prev->list_next = &linked_list->ghost_element;
    item->list_prev = item;
    item->list_next = item;
    dplasma_atomic_unlock(&(linked_list->atomic_lock));

    return (item == &(linked_list->ghost_element) ? NULL : item);
}

#endif  /* DPLASMA_LINKED_LIST_H_HAS_BEEN_INCLUDED */
