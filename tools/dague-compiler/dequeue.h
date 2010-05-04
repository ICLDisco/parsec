/*                                                                                                                                                                    
 * Copyright (c) 2009      The University of Tennessee and The University                                                                                             
 *                         of Tennessee Research Foundation.  All rights                                                                                              
 *                         reserved.                                                                                                                                  
 */

#ifndef DEQUEUE_H_HAS_BEEN_INCLUDED
#define DEQUEUE_H_HAS_BEEN_INCLUDED

#include "atomic.h"
#include "lifo.h"

typedef struct DAGuE_dequeue_t {
    DAGuE_list_item_t  ghost_element;
    uint32_t atomic_lock;
} DAGuE_dequeue_t;

static inline void DAGuE_atomic_lock( volatile uint32_t* atomic_lock )
{
    while( !DAGuE_atomic_cas( atomic_lock, 0, 1) )
        /* nothing */;
}

static inline void DAGuE_atomic_unlock( volatile uint32_t* atomic_lock )
{
    *atomic_lock = 0;
}

static inline int DAGuE_atomic_trylock( volatile uint32_t* atomic_lock )
{
    return DAGuE_atomic_cas( atomic_lock, 0, 1);
}

static inline void DAGuE_dequeue_construct( DAGuE_dequeue_t* dequeue )
{
    dequeue->ghost_element.list_next = &(dequeue->ghost_element);
    dequeue->ghost_element.list_prev = &(dequeue->ghost_element);
    dequeue->atomic_lock = 0;
}

static inline void dplamsa_dequeue_item_construct( DAGuE_list_item_t *item )
{
    item->list_prev = item;
}

static inline int DAGuE_dequeue_is_empty( DAGuE_dequeue_t * dequeue )
{
    int res;
    
    DAGuE_atomic_lock(&(dequeue->atomic_lock));
    
    res = (dequeue->ghost_element.list_prev == &(dequeue->ghost_element)) 
       && (dequeue->ghost_element.list_next == &(dequeue->ghost_element));
    
    DAGuE_atomic_unlock(&(dequeue->atomic_lock));
    return res;
}

static inline DAGuE_list_item_t* DAGuE_dequeue_pop_back( DAGuE_dequeue_t* dequeue )
{
    DAGuE_list_item_t* item;

    if( !DAGuE_atomic_trylock(&(dequeue->atomic_lock)) ) {
        return NULL;
    }

    item = (DAGuE_list_item_t*)dequeue->ghost_element.list_prev;
    dequeue->ghost_element.list_prev = item->list_prev;
    item->list_prev->list_next = &(dequeue->ghost_element);
    
    DAGuE_atomic_unlock(&(dequeue->atomic_lock));

    if( &(dequeue->ghost_element) == item )
        return NULL;

    return item;
}

static inline DAGuE_list_item_t* DAGuE_dequeue_pop_front( DAGuE_dequeue_t* dequeue )
{
    DAGuE_list_item_t* item;

    if( !DAGuE_atomic_trylock(&(dequeue->atomic_lock)) ) {
        return NULL;
    }

    item = (DAGuE_list_item_t*)dequeue->ghost_element.list_next;
    dequeue->ghost_element.list_next = item->list_next;
    item->list_next->list_prev = &(dequeue->ghost_element);
    
    DAGuE_atomic_unlock(&(dequeue->atomic_lock));

    if( &(dequeue->ghost_element) == item )
        return NULL;
    return item;
}

static inline void DAGuE_dequeue_push_back(DAGuE_dequeue_t* dequeue, DAGuE_list_item_t* items )
{
    DAGuE_list_item_t* tail = (DAGuE_list_item_t*)items->list_prev;

    tail->list_next = &(dequeue->ghost_element);

    DAGuE_atomic_lock(&(dequeue->atomic_lock));

    items->list_prev = dequeue->ghost_element.list_prev;
    items->list_prev->list_next = items;
    dequeue->ghost_element.list_prev = tail;

    DAGuE_atomic_unlock(&(dequeue->atomic_lock));
}

static inline void DAGuE_dequeue_push_front(DAGuE_dequeue_t* dequeue, DAGuE_list_item_t* items )
{
    DAGuE_list_item_t* tail = (DAGuE_list_item_t*)items->list_prev;

    items->list_prev = &(dequeue->ghost_element);

    DAGuE_atomic_lock(&(dequeue->atomic_lock));

    tail->list_next = dequeue->ghost_element.list_next;
    tail->list_next->list_prev = tail;
    dequeue->ghost_element.list_next = items;

    DAGuE_atomic_unlock(&(dequeue->atomic_lock));
}

#endif  /* DEQUEUE_H_HAS_BEEN_INCLUDED */
