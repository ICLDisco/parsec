/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef LIFO_H_HAS_BEEN_INCLUDED
#define LIFO_H_HAS_BEEN_INCLUDED

#include "atomic.h"

typedef struct DAGuE_list_item_t {
    volatile struct DAGuE_list_item_t* list_next;
    void* cache_friendly_emptiness;
    volatile struct DAGuE_list_item_t* list_prev;
#ifdef DAGuE_DEBUG
    volatile int32_t refcount;
    volatile struct DAGuE_list_t* belong_to_list;
#endif  /* DAGuE_DEBUG */
} DAGuE_list_item_t;

/* Make a well formed singleton list with a list item so that it can be 
 * puhsed. 
 */
#define DAGuE_LIST_ITEM_SINGLETON(item) DAGuE_list_item_singleton((DAGuE_list_item_t*) item)
static inline DAGuE_list_item_t* DAGuE_list_item_singleton(DAGuE_list_item_t* item)
{
    return (DAGuE_list_item_t*) (item->list_next = item->list_prev = item);
}

typedef struct DAGuE_atomic_lifo_t {
    DAGuE_list_item_t* lifo_head;
    DAGuE_list_item_t  lifo_ghost;
} DAGuE_atomic_lifo_t;

/* The ghost pointer will never change. The head will change via an
 * atomic compare-and-swap. On most architectures the reading of a
 * pointer is an atomic operation so we don't have to protect it.
 */
static inline int DAGuE_atomic_lifo_is_empty( DAGuE_atomic_lifo_t* lifo )
{
    return (lifo->lifo_head == &(lifo->lifo_ghost) ? 1 : 0);
}

/* Add double-linked element to the LIFO. We will return the last head
 * of the list to allow the upper level to detect if this element is
 * the first one in the list (if the list was empty before this
 * operation).
 */
static inline DAGuE_list_item_t* DAGuE_atomic_lifo_push( DAGuE_atomic_lifo_t* lifo,
                                                             DAGuE_list_item_t* items )
{
    DAGuE_list_item_t* tail = (DAGuE_list_item_t*)items->list_prev;
#ifdef DAGuE_DEBUG
    {
        DAGuE_list_item_t* item = items;
        do {
            item->refcount++;
            item->belong_to_list = (struct DAGuE_list_t*)lifo;
            item = (DAGuE_list_item_t*)item->list_next;
        } while (item != tail);
    }
#endif  /* DAGuE_DEBUG */
    do {
        tail->list_next = lifo->lifo_head;
        if( DAGuE_atomic_cas( &(lifo->lifo_head),
                                (uintptr_t) tail->list_next,
                                (uintptr_t) items ) ) {
            return (DAGuE_list_item_t*)tail->list_next;
        }
        /* DO some kind of pause to release the bus */
    } while( 1 );
}

/* Retrieve one element from the LIFO. If we reach the ghost element then the LIFO
 * is empty so we return NULL.
 */
static inline DAGuE_list_item_t* DAGuE_atomic_lifo_pop( DAGuE_atomic_lifo_t* lifo )
{
    DAGuE_list_item_t* item;

    while((item = lifo->lifo_head) != &(lifo->lifo_ghost))
    {
        if( DAGuE_atomic_cas( &(lifo->lifo_head),
                                (uintptr_t) item,
                                (uintptr_t) item->list_next ) )
            break;
        /* Do some kind of pause to release the bus */
    }

    if( item == &(lifo->lifo_ghost) ) return NULL;
#ifdef DAGuE_DEBUG
    item->list_next = NULL;
    item->refcount--;
    item->belong_to_list = NULL;
#endif  /* DAGuE_DEBUG */
    return item;
}

static inline void DAGuE_atomic_lifo_construct( DAGuE_atomic_lifo_t* lifo )
{
    lifo->lifo_ghost.list_next = &(lifo->lifo_ghost);
    lifo->lifo_head = &(lifo->lifo_ghost);
}

#endif  /* LIFO_H_HAS_BEEN_INCLUDED */
