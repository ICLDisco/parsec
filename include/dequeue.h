/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DEQUEUE_H_HAS_BEEN_INCLUDED
#define DEQUEUE_H_HAS_BEEN_INCLUDED

/* DEQUEUE definition. Although the current implementation is a pure
 * remap to the list (see list.h), it is not garanteed as such. If one
 * needs to use both DEQUEUE and non DEQUEUE access, list.h contains
 * convenience functions to emulate a dequeue, that is garanteed to be
 * compatible with list accessors.
 */

#include "dague_config.h"
#include "list.h"

typedef dague_list_t dague_dequeue_t;
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_dequeue_t);

static inline int
dague_dequeue_is_empty( dague_dequeue_t* dequeue ) {
    return dague_list_is_empty((dague_list_t*)dequeue);
}

static inline dague_list_item_t*
dague_dequeue_pop_back( dague_dequeue_t* dequeue ) {
    return dague_list_pop_back((dague_list_t*)dequeue);
}

static inline dague_list_item_t*
dague_dequeue_try_pop_back( dague_dequeue_t* dequeue ) {
    return dague_list_try_pop_back((dague_list_t*)dequeue);
}

static inline dague_list_item_t*
dague_dequeue_pop_front( dague_dequeue_t* dequeue ) {
    return dague_list_pop_front((dague_list_t*)dequeue);
}

static inline dague_list_item_t*
dague_dequeue_try_pop_front( dague_dequeue_t* dequeue ) {
    return dague_list_try_pop_front((dague_list_t*)dequeue);
}

static inline void
dague_dequeue_push_back( dague_dequeue_t* dequeue, dague_list_item_t* item ) {
    dague_list_push_back((dague_list_t*)dequeue, item);
}

static inline void
dague_dequeue_push_front( dague_dequeue_t* dequeue, dague_list_item_t* item ) {
    dague_list_push_front((dague_list_t*)dequeue, item);
}

static inline void
dague_dequeue_chain_front( dague_dequeue_t* dequeue, dague_list_item_t* items ) {
    dague_list_chain_front((dague_list_t*)dequeue, items);
}

static inline void
dague_dequeue_chain_back( dague_dequeue_t* dequeue, dague_list_item_t* items ) {
    dague_list_chain_back((dague_list_t*)dequeue, items);
}

static inline int
dague_dequeue_nolock_is_empty( dague_dequeue_t* dequeue) {
    return dague_list_nolock_is_empty((dague_list_t*)dequeue);
}
#define dague_udequeue_is_empty(dequeue) dague_dequeue_nolock_is_empty(dequeue)

static inline dague_list_item_t*
dague_dequeue_nolock_pop_front( dague_dequeue_t* dequeue ) {
    return dague_list_nolock_pop_front((dague_list_t*)dequeue);
}
#define dague_udequeue_pop_front(dequeue) dague_dequeue_nolock_pop_front(dequeue)

static inline dague_list_item_t*
dague_dequeue_nolock_pop_back( dague_dequeue_t* dequeue ) {
    return dague_list_nolock_pop_back((dague_list_t*)dequeue);
}
#define dague_udequeue_pop_back(dequeue) dague_dequeue_nolock_pop_back(dequeue)

static inline void
dague_dequeue_nolock_push_front( dague_dequeue_t* dequeue, dague_list_item_t* item ) {
    dague_list_nolock_push_front((dague_list_t*)dequeue, item);
}
#define dague_udequeue_push_front(dequeue, item) dague_dequeue_nolock_push_front(dequeue, item)

static inline void
dague_dequeue_nolock_push_back( dague_dequeue_t* dequeue, dague_list_item_t* item ) {
    dague_list_nolock_push_back((dague_list_t*)dequeue, item);
}
#define dague_udequeue_push_back(dequeue, item) dague_dequeue_nolock_push_back(dequeue, item)

static inline void
dague_dequeue_nolock_chain_front( dague_dequeue_t* dequeue, dague_list_item_t* items ) {
    dague_list_nolock_chain_front((dague_list_t*)dequeue, items);
}
#define dague_udequeue_chainf(dequeue, items) dague_dequeue_nolock_chain_front(dequeue, items)

static inline void
dague_dequeue_nolock_chain_back( dague_dequeue_t* dequeue, dague_list_item_t* items ) {
    dague_list_nolock_chain_back((dague_list_t*)dequeue, items);
}
#define dague_udequeue_chain_back(dequeue, items) dague_dequeue_nolock_chain_back(dequeue, items)


#endif  /* DEQUEUE_H_HAS_BEEN_INCLUDED */
