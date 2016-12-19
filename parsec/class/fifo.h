/*
 * Copyright (c) 2010-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef FIFO_H_HAS_BEEN_INCLUDED
#define FIFO_H_HAS_BEEN_INCLUDED

/* FIFO definition. Although the current implementation is a pure remap
 * to the list (see list.h), it is not garanteed as such. If you need to
 * use both FIFO and non FIFO access, list.h contains convenience
 * functions to emulate a fifo that is garanteed to be compatible with list accessors.
 */

#include "parsec_config.h"
#include "parsec/class/list.h"

typedef parsec_list_t parsec_fifo_t;
PARSEC_DECLSPEC OBJ_CLASS_DECLARATION(parsec_fifo_t);

static inline int
parsec_fifo_is_empty( parsec_fifo_t* fifo ) {
    return parsec_list_is_empty((parsec_list_t*)fifo);
}

static inline int
parsec_fifo_nolock_is_empty( parsec_fifo_t* fifo)
{
    return parsec_list_nolock_is_empty((parsec_list_t*)fifo);
}

static inline void
parsec_fifo_push(parsec_fifo_t* fifo, parsec_list_item_t* item) {
    parsec_list_push_front((parsec_list_t*)fifo, item);
}
static inline void
parsec_fifo_nolock_push(parsec_fifo_t* fifo, parsec_list_item_t* item) {
    parsec_list_nolock_push_front((parsec_list_t*)fifo, item);
}

static inline void
parsec_fifo_chain(parsec_fifo_t* fifo, parsec_list_item_t* items) {
    parsec_list_chain_front((parsec_list_t*)fifo, items);
}
static inline void
parsec_fifo_nolock_chain(parsec_fifo_t* fifo, parsec_list_item_t* items) {
    parsec_list_nolock_chain_front((parsec_list_t*)fifo, items);
}

static inline parsec_list_item_t*
parsec_fifo_pop(parsec_fifo_t* fifo) {
    return parsec_list_pop_front((parsec_list_t*)fifo);
}
static inline parsec_list_item_t*
parsec_fifo_try_pop(parsec_fifo_t* fifo) {
    return parsec_list_try_pop_front((parsec_list_t*)fifo);
}
static inline parsec_list_item_t*
parsec_fifo_nolock_pop(parsec_fifo_t* fifo) {
    return parsec_list_nolock_pop_front((parsec_list_t*)fifo);
}

#endif  /* FIFO_H_HAS_BEEN_INCLUDED */
