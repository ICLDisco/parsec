/*
 * Copyright (c) 2010      The University of Tennessee and The University
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
#include "list.h"

typedef dague_list_t dague_fifo_t;

static inline void 
dague_fifo_construct( dague_fifo_t* fifo ) {
    dague_list_construct((dague_list_t*)fifo);
}

static inline void
dague_fifo_destruct( dague_fifo_t* fifo ) {
    dague_list_destruct((dague_list_t*)fifo);
}

static inline int
dague_fifo_is_empty( dague_fifo_t* fifo ) {
    return dague_list_is_empty((dague_list_t*)fifo);
}

static inline int 
dague_fifo_nolock_is_empty( dague_fifo_t* fifo)
{
    return dague_list_nolock_is_empty((dague_list_t*)fifo);
}
#define dague_ufifo_is_empty(fifo) dague_fifo_nolock_is_empty(fifo)

static inline void
dague_fifo_push(dague_list_t* list, dague_list_item_t* item) {
    dague_list_push_back(list, item); 
}
static inline void
dague_fifo_nolock_push(dague_list_t* list, dague_list_item_t* item) { 
    dague_list_nolock_push_back(list, item); 
}
#define dague_ufifo_push(list, item) dague_fifo_nolock_push(list, item)

static inline void
dague_fifo_chain(dague_list_t* list, dague_list_item_t* items) {
    dague_list_chain_back(list, items);
}
static inline void
dague_fifo_nolock_chain(dague_list_t* list, dague_list_item_t* items) { 
    dague_list_nolock_chain_back(list, items);
}
#define dague_ufifo_chain(list, items) dague_fifo_nolock_chain(list, items)

static inline dague_list_item_t*
dague_fifo_pop(dague_list_t* list) {
    return dague_list_pop_front(list); 
}
static inline dague_list_item_t*
dague_fifo_try_pop(dague_list_t* list) {
    return dague_list_try_pop_front(list);
}
#define dague_fifo_tpop(list) dague_fifo_try_pop(list)
static inline dague_list_item_t* 
dague_fifo_nolock_pop(dague_list_t* list) { 
    return dague_list_nolock_pop_front(list); 
}
#define dague_ufifo_pop(list) dague_fifo_nolock_pop(list)

#endif  /* FIFO_H_HAS_BEEN_INCLUDED */
