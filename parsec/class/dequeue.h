/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
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

#include "parsec/parsec_config.h"
#include "parsec/class/list.h"

BEGIN_C_DECLS

typedef parsec_list_t parsec_dequeue_t;
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_dequeue_t);

static inline int
parsec_dequeue_is_empty( parsec_dequeue_t* dequeue ) {
    return parsec_list_is_empty((parsec_list_t*)dequeue);
}

static inline parsec_list_item_t*
parsec_dequeue_pop_back( parsec_dequeue_t* dequeue ) {
    return parsec_list_pop_back((parsec_list_t*)dequeue);
}

static inline parsec_list_item_t*
parsec_dequeue_try_pop_back( parsec_dequeue_t* dequeue ) {
    return parsec_list_try_pop_back((parsec_list_t*)dequeue);
}

static inline parsec_list_item_t*
parsec_dequeue_pop_front( parsec_dequeue_t* dequeue ) {
    return parsec_list_pop_front((parsec_list_t*)dequeue);
}

static inline parsec_list_item_t*
parsec_dequeue_try_pop_front( parsec_dequeue_t* dequeue ) {
    return parsec_list_try_pop_front((parsec_list_t*)dequeue);
}

static inline void
parsec_dequeue_push_back( parsec_dequeue_t* dequeue, parsec_list_item_t* item ) {
    parsec_list_push_back((parsec_list_t*)dequeue, item);
}

static inline void
parsec_dequeue_push_front( parsec_dequeue_t* dequeue, parsec_list_item_t* item ) {
    parsec_list_push_front((parsec_list_t*)dequeue, item);
}

static inline void
parsec_dequeue_chain_front( parsec_dequeue_t* dequeue, parsec_list_item_t* items ) {
    parsec_list_chain_front((parsec_list_t*)dequeue, items);
}

static inline void
parsec_dequeue_chain_back( parsec_dequeue_t* dequeue, parsec_list_item_t* items ) {
    parsec_list_chain_back((parsec_list_t*)dequeue, items);
}

static inline int
parsec_dequeue_nolock_is_empty( parsec_dequeue_t* dequeue) {
    return parsec_list_nolock_is_empty((parsec_list_t*)dequeue);
}
#define parsec_udequeue_is_empty(dequeue) parsec_dequeue_nolock_is_empty(dequeue)

static inline parsec_list_item_t*
parsec_dequeue_nolock_pop_front( parsec_dequeue_t* dequeue ) {
    return parsec_list_nolock_pop_front((parsec_list_t*)dequeue);
}
#define parsec_udequeue_pop_front(dequeue) parsec_dequeue_nolock_pop_front(dequeue)

static inline parsec_list_item_t*
parsec_dequeue_nolock_pop_back( parsec_dequeue_t* dequeue ) {
    return parsec_list_nolock_pop_back((parsec_list_t*)dequeue);
}
#define parsec_udequeue_pop_back(dequeue) parsec_dequeue_nolock_pop_back(dequeue)

static inline void
parsec_dequeue_nolock_push_front( parsec_dequeue_t* dequeue, parsec_list_item_t* item ) {
    parsec_list_nolock_push_front((parsec_list_t*)dequeue, item);
}
#define parsec_udequeue_push_front(dequeue, item) parsec_dequeue_nolock_push_front(dequeue, item)

static inline void
parsec_dequeue_nolock_push_back( parsec_dequeue_t* dequeue, parsec_list_item_t* item ) {
    parsec_list_nolock_push_back((parsec_list_t*)dequeue, item);
}
#define parsec_udequeue_push_back(dequeue, item) parsec_dequeue_nolock_push_back(dequeue, item)

static inline void
parsec_dequeue_nolock_chain_front( parsec_dequeue_t* dequeue, parsec_list_item_t* items ) {
    parsec_list_nolock_chain_front((parsec_list_t*)dequeue, items);
}
#define parsec_udequeue_chainf(dequeue, items) parsec_dequeue_nolock_chain_front(dequeue, items)

static inline void
parsec_dequeue_nolock_chain_back( parsec_dequeue_t* dequeue, parsec_list_item_t* items ) {
    parsec_list_nolock_chain_back((parsec_list_t*)dequeue, items);
}
#define parsec_udequeue_chain_back(dequeue, items) parsec_dequeue_nolock_chain_back(dequeue, items)

END_C_DECLS

#endif  /* DEQUEUE_H_HAS_BEEN_INCLUDED */
