/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* This file contains functions to access simple lists and stacks, 
 * When thread safe locking performance is critical, one could prefer 
 * fifo, lifo or dequeues
 */

#ifndef DAGUE_LIST_H_HAS_BEEN_INCLUDED
#define DAGUE_LIST_H_HAS_BEEN_INCLUDED

#include "atomic.h"
#include "list_item.h"

typedef struct dague_list_t {
    dague_list_item_t  ghost_element;
    uint32_t atomic_lock;
} dague_list_t;

static inline void 
dague_list_construct( dague_list_t* linked_list )
{
    linked_list->ghost_element.list_next = &(linked_list->ghost_element);
    linked_list->ghost_element.list_prev = &(linked_list->ghost_element);
    linked_list->atomic_lock = 0;
}

static inline int 
dague_list_is_empty( dague_list_t * linked_list )
{
    return linked_list->ghost_element.list_next != &(linked_list->ghost_element);
}


/* Add/remove items from a list. nolock version do not lock the list */
static inline void
dague_list_nolock_add_head( dague_list_t* linked_list, 
                            dague_list_item_t* item )
{
    item->list_prev = &(linked_list->ghost_element);
    item->list_next = linked_list->ghost_element.list_next;
    linked_list->ghost_element.list_next->list_prev = item;
    linked_list->ghost_element.list_next = item;
    DAGUE_ATTACH_ELEM(linked_list, item);                                
}

static inline void 
dague_list_add_head( dague_list_t * linked_list,
                     dague_list_item_t *item )
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    dague_list_nolock_add_head(linked_list, item);
    dague_atomic_unlock(&(linked_list->atomic_lock));
}


static inline void 
dague_list_nolock_add_tail( dague_list_t * linked_list,
                            dague_list_item_t *item )
{
    item->list_next = &(linked_list->ghost_element);
    item->list_prev = linked_list->ghost_element.list_prev;
    linked_list->ghost_element.list_prev->list_next = item;
    linked_list->ghost_element.list_prev = item;
    DAGUE_ATTACH_ELEM(linked_list, item);
}

static inline void 
dague_list_add_tail( dague_list_t * linked_list,
                     dague_list_item_t *item )
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    dague_list_nolock_add_tail(linked_list, item);
    dague_atomic_unlock(&(linked_list->atomic_lock));
}


static inline dague_list_item_t*
dague_list_nolock_remove_head( dague_list_t * linked_list )
{
    dague_list_item_t* item;

    item = (dague_list_item_t*)linked_list->ghost_element.list_next;
    linked_list->ghost_element.list_next = item->list_next;
    item->list_next->list_prev = &linked_list->ghost_element;
    item->list_prev = item;
    item->list_next = item;
    if( &(linked_list->ghost_element) != item ) {
        DAGUE_DETACH_ELEM(item);
        return item;
    }
    return NULL;
}

static inline dague_list_item_t*
dague_list_remove_head( dague_list_t * linked_list )
{
    dague_list_item_t* item;
    dague_atomic_lock(&(linked_list->atomic_lock));
    item = dague_list_nolock_remove_head(linked_list);
    dague_atomic_unlock(&(linked_list->atomic_lock));
    return item;
}


static inline dague_list_item_t*
dague_list_nolock_remove_tail( dague_list_t * linked_list )
{
    dague_list_item_t* item;

    item = (dague_list_item_t*)linked_list->ghost_element.list_prev;
    linked_list->ghost_element.list_prev = item->list_prev;
    item->list_prev->list_next = &linked_list->ghost_element;
    item->list_prev = item;
    item->list_next = item;
    if( &(linked_list->ghost_element) != item ) {
        DAGUE_DETACH_ELEM(item);
        return item;
    }
    return NULL;
}

static inline dague_list_item_t*
dague_list_remove_tail( dague_list_t * linked_list )
{
    dague_list_item_t* item;
    dague_atomic_lock(&(linked_list->atomic_lock));
    item = dague_list_nolock_remove_tail(linked_list);
    dague_atomic_unlock(&(linked_list->atomic_lock));
    return item;
}


static inline dague_list_item_t*
dague_list_nolock_remove_item( dague_list_t * linked_list,
                               dague_list_item_t* item)
{
#if defined(DAGUE_DEBUG)
    assert(item->belong_to_list == linked_list);
#else
    (void)linked_list;
#endif
    item->list_prev->list_next = item->list_next;
    item->list_next->list_prev = item->list_prev;
    item->list_next = item;
    item->list_prev = item;
    DAGUE_DETACH_ELEM(item);
    return item;
}

static inline dague_list_item_t*
dague_list_remove_item( dague_list_t * linked_list,
                        dague_list_item_t* item)
{
    dague_atomic_lock(&(linked_list->atomic_lock));
    item = dague_list_nolock_remove_item(linked_list, item);
    dague_atomic_unlock(&(linked_list->atomic_lock));
    return item;
}

/* define some convenience function shortnames */
/* uop versions are not locked versions of op */
#define dague_list_upush(list, item) dague_list_nolock_add_head(list, item)
#define dague_list_push(list, item) dague_list_add_head(list, item)
#define dague_list_upop(list) dague_list_nolock_remove_head(list)
#define dague_list_pop(list) dague_list_remove_head(list)


/* Iterate functions permits traversing the list. The considered items 
 *   remain in the list. Hence, the iterate functions are not thread 
 *   safe. If the list or the list items are modified (even with 
 *   locked remove/add), status is undetermined.
 * Typical use: 
 *   for( item = iterate_head(list); 
 *        item != NULL; 
 *        item = iterate_next(list, item) ) { use_item(item); }
 *  "use_item()"" should not change item->list_next, item->list_prev
 */
static inline dague_list_item_t*
dague_list_iterate_head( dague_list_t* linked_list )
{
    if(linked_list->ghost_element.list_next == &(linked_list->ghost_element))
        return NULL;
    return (dague_list_item_t*) linked_list->ghost_element.list_next;
}

static inline dague_list_item_t*
dague_list_iterate_tail( dague_list_t* linked_list )
{
    if(linked_list->ghost_element.list_prev == &(linked_list->ghost_element))
        return NULL;
    return (dague_list_item_t*) linked_list->ghost_element.list_prev;
}

static inline dague_list_item_t*
dague_list_iterate_next( dague_list_t* linked_list, 
                         dague_list_item_t* item )
{
    if(item->list_next == &(linked_list->ghost_element)) return NULL;
    else return (dague_list_item_t*) item->list_next;
}

static inline dague_list_item_t*
dague_list_iterate_prev( dague_list_t* linked_list, 
                         dague_list_item_t* item )
{
    if(item->list_prev == &(linked_list->ghost_element)) return NULL;
    else return (dague_list_item_t*) item->list_prev;
}

/* Remove current item, and returns the next */
static inline dague_list_item_t*
dague_list_iterate_remove_and_next( dague_list_t* linked_list, 
                                    dague_list_item_t* item )
{
    dague_list_item_t* next = dague_list_iterate_next(linked_list, item);
    dague_list_nolock_remove_item(linked_list, item);
    return next;
}

static inline dague_list_item_t*
dague_list_iterate_remove_and_prev( dague_list_t* linked_list,
                                    dague_list_item_t* item )
{
    dague_list_item_t* prev = dague_list_iterate_prev(linked_list, item);
    dague_list_nolock_remove_item(linked_list, item);
    return prev;
}

#define dague_list_iterate_remove(L,I) dague_list_iterate_remove_and_next(L,I)

/* Lock the list while it is used, to prevent concurent add/remove */
static inline void 
dague_list_iterate_lock( dague_list_t* linked_list )
{
    dague_atomic_lock(&(linked_list->atomic_lock));
}

static inline void
dague_list_iterate_unlock( dague_list_t* linked_list )
{
    dague_atomic_unlock(&(linked_list->atomic_lock));
}



#endif  /* DAGUE_LIST_H_HAS_BEEN_INCLUDED */
