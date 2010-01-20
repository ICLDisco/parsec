/*                                                                                                                                                                    
 * Copyright (c) 2009      The University of Tennessee and The University                                                                                             
 *                         of Tennessee Research Foundation.  All rights                                                                                              
 *                         reserved.                                                                                                                                  
 */

#ifndef DEQUEUE_H_HAS_BEEN_INCLUDED
#define DEQUEUE_H_HAS_BEEN_INCLUDED

#include "atomic.h"
#include "lifo.h"

typedef struct dplasma_dequeue_t {
    dplasma_list_item_t  ghost_element;
    uint32_t atomic_lock;
} dplasma_dequeue_t;

static inline void dplasma_atomic_lock( volatile uint32_t* atomic_lock )
{
    while( !dplasma_atomic_cas( atomic_lock, 0, 1) )
        /* nothing */;
}

static inline void dplasma_atomic_unlock( volatile uint32_t* atomic_lock )
{
    *atomic_lock = 0;
}

static inline int dplasma_atomic_trylock( volatile uint32_t* atomic_lock )
{
    return dplasma_atomic_cas( atomic_lock, 0, 1);
}

static inline void dplasma_dequeue_construct( dplasma_dequeue_t* dequeue )
{
    dequeue->ghost_element.list_next = &(dequeue->ghost_element);
    dequeue->ghost_element.list_prev = &(dequeue->ghost_element);
    dequeue->atomic_lock = 0;
}

static inline int dplasma_dequeue_is_empty( dplasma_dequeue_t * dequeue )
{
    int res;
    
    dplasma_atomic_lock(&(dequeue->atomic_lock));
    
    res = (dequeue->ghost_element.list_prev == &(dequeue->ghost_element)) 
       && (dequeue->ghost_element.list_next == &(dequeue->ghost_element));
    
    dplasma_atomic_unlock(&(dequeue->atomic_lock));
    return res;
}

static inline dplasma_list_item_t* dplasma_dequeue_pop_back( dplasma_dequeue_t* dequeue )
{
    dplasma_list_item_t* item;

    if( !dplasma_atomic_trylock(&(dequeue->atomic_lock)) ) {
        return NULL;
    }

    item = (dplasma_list_item_t*)dequeue->ghost_element.list_prev;
    dequeue->ghost_element.list_prev = item->list_prev;
    item->list_prev->list_next = &(dequeue->ghost_element);
    
    dplasma_atomic_unlock(&(dequeue->atomic_lock));

    if( &(dequeue->ghost_element) == item )
        return NULL;

    return item;
}

static inline dplasma_list_item_t* dplasma_dequeue_pop_front( dplasma_dequeue_t* dequeue )
{
    dplasma_list_item_t* item;

    if( !dplasma_atomic_trylock(&(dequeue->atomic_lock)) ) {
        return NULL;
    }

    item = (dplasma_list_item_t*)dequeue->ghost_element.list_next;
    dequeue->ghost_element.list_next = item->list_next;
    item->list_next->list_prev = &(dequeue->ghost_element);
    
    dplasma_atomic_unlock(&(dequeue->atomic_lock));

    if( &(dequeue->ghost_element) == item )
        return NULL;
    return item;
}

static inline void dplasma_dequeue_push_back(dplasma_dequeue_t* dequeue, dplasma_list_item_t* item )
{
    item->list_next = &(dequeue->ghost_element);

    dplasma_atomic_lock(&(dequeue->atomic_lock));

    item->list_prev = dequeue->ghost_element.list_prev;
    item->list_prev->list_next = item;
    dequeue->ghost_element.list_prev = item;

    dplasma_atomic_unlock(&(dequeue->atomic_lock));
}

static inline void dplasma_dequeue_push_front(dplasma_dequeue_t* dequeue, dplasma_list_item_t* item )
{
    item->list_prev = &(dequeue->ghost_element);

    dplasma_atomic_lock(&(dequeue->atomic_lock));

    item->list_next = dequeue->ghost_element.list_next;
    item->list_next->list_prev = item;
    dequeue->ghost_element.list_next = item;

    dplasma_atomic_unlock(&(dequeue->atomic_lock));
}

#endif  /* DEQUEUE_H_HAS_BEEN_INCLUDED */
