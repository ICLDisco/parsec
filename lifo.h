/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef LIFO_H_HAS_BEEN_INCLUDED
#define LIFO_H_HAS_BEEN_INCLUDED

#include "atomic.h"

typedef struct dplasma_list_item_t {
    volatile struct dplasma_list_item_t* list_next;
    volatile struct dplasma_list_item_t* list_prev;
#ifdef DPLASMA_DEBUG
    volatile int32_t refcount;
    volatile struct dplasma_list_t* belong_to_list;
#endif  /* DPLASMA_DEBUG */
} dplasma_list_item_t;

typedef struct dplasma_atomic_lifo_t {
    dplasma_list_item_t* lifo_head;
    dplasma_list_item_t  lifo_ghost;
} dplasma_atomic_lifo_t;

/* The ghost pointer will never change. The head will change via an atomic
 * compare-and-swap. On most architectures the reading of a pointer is an
 * atomic operation so we don't have to protect it.
 */
static inline int dplasma_atomic_lifo_is_empty( dplasma_atomic_lifo_t* lifo )
{
    return (lifo->lifo_head == &(lifo->lifo_ghost) ? 1 : 0);
}

/* Add one element to the LIFO. We will return the last head of the list
 * to allow the upper level to detect if this element is the first one in the
 * list (if the list was empty before this operation).
 */
static inline dplasma_list_item_t* dplasma_atomic_lifo_push( dplasma_atomic_lifo_t* lifo,
                                                             dplasma_list_item_t* item )
{
#ifdef DPLASMA_DEBUG
    item->refcount++;
    item->belong_to_list = (struct dplasma_list_t*)lifo;
#endif  /* DPLASMA_DEBUG */
    do {
        item->list_next = lifo->lifo_head;
        if( dplasma_atomic_cas( &(lifo->lifo_head),
                                (void*)item->list_next,
                                item ) ) {
            return (dplasma_list_item_t*)item->list_next;
        }
        /* DO some kind of pause to release the bus */
    } while( 1 );
}

/* Retrieve one element from the LIFO. If we reach the ghost element then the LIFO
 * is empty so we return NULL.
 */
static inline dplasma_list_item_t* dplasma_atomic_lifo_pop( dplasma_atomic_lifo_t* lifo )
{
    dplasma_list_item_t* item;

    while((item = lifo->lifo_head) != &(lifo->lifo_ghost))
    {
        if( dplasma_atomic_cas( &(lifo->lifo_head),
                                item,
                                (void*)item->list_next ) )
            break;
        /* Do some kind of pause to release the bus */
    }

    if( item == &(lifo->lifo_ghost) ) return NULL;
#ifdef DPLASMA_DEBUG
    item->list_next = NULL;
    item->refcount--;
    item->belong_to_list = NULL;
#endif  /* DPLASMA_DEBUG */
    return item;
}

static inline void dplasma_atomic_lifo_construct( dplasma_atomic_lifo_t* lifo )
{
    lifo->lifo_ghost.list_next = &(lifo->lifo_ghost);
    lifo->lifo_head = &(lifo->lifo_ghost);
}

#endif  /* LIFO_H_HAS_BEEN_INCLUDED */
