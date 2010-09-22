/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef LIFO_H_HAS_BEEN_INCLUDED
#define LIFO_H_HAS_BEEN_INCLUDED

#include "atomic.h"

typedef struct dague_list_item_t {
    volatile struct dague_list_item_t* list_next;
    void* cache_friendly_emptiness;
    volatile struct dague_list_item_t* list_prev;
#ifdef DAGUE_DEBUG
    volatile int32_t refcount;
    volatile struct dague_list_t* belong_to_list;
#endif  /* DAGUE_DEBUG */
} dague_list_item_t;

/* Make a well formed singleton list with a list item so that it can be 
 * puhsed. 
 */
#define DAGUE_LIST_ITEM_SINGLETON(item) dague_list_item_singleton((dague_list_item_t*) item)
static inline dague_list_item_t* dague_list_item_singleton(dague_list_item_t* item)
{
    item->list_next = item;
    item->list_prev = item;
    return item;
}


#if defined(DAGUE_LIFO_USE_LOCKS)

typedef struct dague_atomic_lifo_t {
    volatile dague_list_item_t* lifo_head;
    dague_list_item_t  lifo_ghost;
    volatile uint32_t  lifo_lock;
} dague_atomic_lifo_t;

static inline int dague_atomic_lifo_is_empty( dague_atomic_lifo_t* lifo )
{
    int ret;
    dague_atomic_lock( &lifo->lifo_lock );
    ret = (lifo->lifo_head == &(lifo->lifo_ghost) ? 1 : 0);
    dague_atomic_unlock( &lifo->lifo_lock );
    return ret;
}

/* Add double-linked element to the LIFO. We will return the last head
 * of the list to allow the upper level to detect if this element is
 * the first one in the list (if the list was empty before this
 * operation).
 */
static inline dague_list_item_t* dague_atomic_lifo_push( dague_atomic_lifo_t* lifo,
                                                         dague_list_item_t* items )
{
    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;
    dague_list_item_t* ret;

#ifdef DAGUE_DEBUG
    {
        dague_list_item_t* item = items;
        do {
            item->refcount++;
            item->belong_to_list = (struct dague_list_t*)lifo;
            item = (dague_list_item_t*)item->list_next;
        } while (item != tail);
    }
#endif  /* DAGUE_DEBUG */

    dague_atomic_lock( &lifo->lifo_lock );
    ret = (dague_list_item_t*)lifo->lifo_head;
    tail->list_next = ret;
    lifo->lifo_head = items;
    dague_atomic_unlock( &lifo->lifo_lock );
    return ret;
}

/* Retrieve one element from the LIFO. If we reach the ghost element then the LIFO
 * is empty so we return NULL.
 */
static inline dague_list_item_t* dague_atomic_lifo_pop( dague_atomic_lifo_t* lifo )
{
    dague_list_item_t* item;

    dague_atomic_lock( &lifo->lifo_lock );
    item = (dague_list_item_t*)lifo->lifo_head;
    lifo->lifo_head = item->list_next;
    dague_atomic_unlock( &lifo->lifo_lock );

    if( item == &(lifo->lifo_ghost) ) 
        return NULL;

#if defined(DAGUE_DEBUG)
    item->refcount--;
    item->belong_to_list = NULL;
#endif /* DAGUE_DEBUG */

    return item;
}

static inline void dague_atomic_lifo_construct( dague_atomic_lifo_t* lifo )
{
    lifo->lifo_ghost.list_next = &(lifo->lifo_ghost);
    lifo->lifo_head = &(lifo->lifo_ghost);
    lifo->lifo_lock = 0;
}


#else


typedef struct dague_atomic_lifo_t {
    volatile dague_list_item_t* lifo_head;
    dague_list_item_t  lifo_ghost;
} dague_atomic_lifo_t;

/* The ghost pointer will never change. The head will change via an
 * atomic compare-and-swap. On most architectures the reading of a
 * pointer is an atomic operation so we don't have to protect it.
 */
static inline int dague_atomic_lifo_is_empty( dague_atomic_lifo_t* lifo )
{
    return (lifo->lifo_head == &(lifo->lifo_ghost) ? 1 : 0);
}

/* Add double-linked element to the LIFO. We will return the last head
 * of the list to allow the upper level to detect if this element is
 * the first one in the list (if the list was empty before this
 * operation).
 */
static inline dague_list_item_t* dague_atomic_lifo_push( dague_atomic_lifo_t* lifo,
                                                             dague_list_item_t* items )
{
    dague_list_item_t* tail = (dague_list_item_t*)items->list_prev;
#ifdef DAGUE_DEBUG
    {
        dague_list_item_t* item = items;
        do {
            item->refcount++;
            item->belong_to_list = (struct dague_list_t*)lifo;
            item = (dague_list_item_t*)item->list_next;
        } while (item != tail);
    }
#endif  /* DAGUE_DEBUG */
    do {
        tail->list_next = lifo->lifo_head;
        dague_mfence();
        if( dague_atomic_cas( &(lifo->lifo_head),
                                (uintptr_t) tail->list_next,
                                (uintptr_t) items ) ) {            
            return (dague_list_item_t*)tail->list_next;
        }
        /* DO some kind of pause to release the bus */
    } while( 1 );
}

/* Retrieve one element from the LIFO. If we reach the ghost element then the LIFO
 * is empty so we return NULL.
 */
static inline dague_list_item_t* dague_atomic_lifo_pop( dague_atomic_lifo_t* lifo )
{
    dague_list_item_t* item;

    while((item = (dague_list_item_t*)lifo->lifo_head) != &(lifo->lifo_ghost))
    {
        if( dague_atomic_cas( &(lifo->lifo_head),
                                (uintptr_t) item,
                                (uintptr_t) item->list_next ) )
            break;
        /* Do some kind of pause to release the bus */
    }

    if( item == &(lifo->lifo_ghost) ) return NULL;
#ifdef DAGUE_DEBUG
    item->list_next = NULL;
    item->refcount--;
    item->belong_to_list = NULL;
#endif  /* DAGUE_DEBUG */
    return item;
}

static inline void dague_atomic_lifo_construct( dague_atomic_lifo_t* lifo )
{
    lifo->lifo_ghost.list_next = &(lifo->lifo_ghost);
    lifo->lifo_head = &(lifo->lifo_ghost);
    dague_mfence();
}

#endif

#endif  /* LIFO_H_HAS_BEEN_INCLUDED */
