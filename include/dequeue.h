/*                                                                                                                                                                    
 * Copyright (c) 2009      The University of Tennessee and The University                                                                                             
 *                         of Tennessee Research Foundation.  All rights                                                                                              
 *                         reserved.                                                                                                                                  
 */

#ifndef DEQUEUE_H_HAS_BEEN_INCLUDED
#define DEQUEUE_H_HAS_BEEN_INCLUDED

#include "atomic.h"
#include "lifo.h"

typedef struct dague_dequeue_t {
    dague_list_item_t  ghost_element;
    uint32_t atomic_lock;
} dague_dequeue_t;

static inline void dague_dequeue_construct( dague_dequeue_t* dequeue )
{
    dequeue->ghost_element.list_next = &(dequeue->ghost_element);
    dequeue->ghost_element.list_prev = &(dequeue->ghost_element);
    dequeue->atomic_lock = 0;
}

static inline void dplamsa_dequeue_item_construct( dague_list_item_t *item )
{
    item->list_prev = item;
}

static inline int dague_dequeue_is_empty( dague_dequeue_t * dequeue )
{
    int res;
    
    dague_atomic_lock(&(dequeue->atomic_lock));
    
    res = (dequeue->ghost_element.list_prev == &(dequeue->ghost_element)) 
       && (dequeue->ghost_element.list_next == &(dequeue->ghost_element));
    
    dague_atomic_unlock(&(dequeue->atomic_lock));
    return res;
}

static inline dague_list_item_t* dague_dequeue_pop_back( dague_dequeue_t* dequeue )
{
    dague_list_item_t* item;

    if( !dague_atomic_trylock(&(dequeue->atomic_lock)) ) {
        return NULL;
    }

    item = (dague_list_item_t*)dequeue->ghost_element.list_prev;
    dequeue->ghost_element.list_prev = item->list_prev;
    item->list_prev->list_next = &(dequeue->ghost_element);
    
    dague_atomic_unlock(&(dequeue->atomic_lock));

    if( &(dequeue->ghost_element) == item )
        return NULL;

    return item;
}

static inline dague_list_item_t* dague_dequeue_pop_front( dague_dequeue_t* dequeue )
{
    dague_list_item_t* item;

    if( !dague_atomic_trylock(&(dequeue->atomic_lock)) ) {
        return NULL;
    }

    item = (dague_list_item_t*)dequeue->ghost_element.list_next;
    dequeue->ghost_element.list_next = item->list_next;
    item->list_next->list_prev = &(dequeue->ghost_element);
    
    dague_atomic_unlock(&(dequeue->atomic_lock));

    if( &(dequeue->ghost_element) == item )
        return NULL;
    return item;
}

static inline void dague_dequeue_push_back(dague_dequeue_t* dequeue, dague_list_item_t* items )
{
    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;

    tail->list_next = &(dequeue->ghost_element);

    dague_atomic_lock(&(dequeue->atomic_lock));

    items->list_prev = dequeue->ghost_element.list_prev;
    items->list_prev->list_next = items;
    dequeue->ghost_element.list_prev = tail;

    dague_atomic_unlock(&(dequeue->atomic_lock));
}

static inline void dague_dequeue_push_front(dague_dequeue_t* dequeue, dague_list_item_t* items )
{
    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;

    items->list_prev = &(dequeue->ghost_element);

    dague_atomic_lock(&(dequeue->atomic_lock));

    tail->list_next = dequeue->ghost_element.list_next;
    tail->list_next->list_prev = tail;
    dequeue->ghost_element.list_next = items;

    dague_atomic_unlock(&(dequeue->atomic_lock));
}

#endif  /* DEQUEUE_H_HAS_BEEN_INCLUDED */
