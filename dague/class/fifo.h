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

#include "dague_config.h"
#include "dague/class/list.h"

typedef dague_list_t dague_fifo_t;
DAGUE_DECLSPEC OBJ_CLASS_DECLARATION(dague_fifo_t);

static inline int
dague_fifo_is_empty( dague_fifo_t* fifo ) {
    return dague_list_is_empty((dague_list_t*)fifo);
}

static inline int
dague_fifo_nolock_is_empty( dague_fifo_t* fifo)
{
    return dague_list_nolock_is_empty((dague_list_t*)fifo);
}

static inline void
dague_fifo_push(dague_fifo_t* fifo, dague_list_item_t* item) {
    dague_list_push_front((dague_list_t*)fifo, item);
}
static inline void
dague_fifo_nolock_push(dague_fifo_t* fifo, dague_list_item_t* item) {
    dague_list_nolock_push_front((dague_list_t*)fifo, item);
}

static inline void
dague_fifo_chain(dague_fifo_t* fifo, dague_list_item_t* items) {
    dague_list_chain_front((dague_list_t*)fifo, items);
}
static inline void
dague_fifo_nolock_chain(dague_fifo_t* fifo, dague_list_item_t* items) {
    dague_list_nolock_chain_front((dague_list_t*)fifo, items);
}

static inline dague_list_item_t*
dague_fifo_pop(dague_fifo_t* fifo) {
    return dague_list_pop_front((dague_list_t*)fifo);
}
static inline dague_list_item_t*
dague_fifo_try_pop(dague_fifo_t* fifo) {
    return dague_list_try_pop_front((dague_list_t*)fifo);
}
static inline dague_list_item_t*
dague_fifo_nolock_pop(dague_fifo_t* fifo) {
    return dague_list_nolock_pop_front((dague_list_t*)fifo);
}

#endif  /* FIFO_H_HAS_BEEN_INCLUDED */
