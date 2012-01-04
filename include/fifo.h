/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef FIFO_H_HAS_BEEN_INCLUDED
#define FIFO_H_HAS_BEEN_INCLUDED

/* FIRST-IN FIRST-OUT operators for lists. Most operators are remaps 
 * to the according list operators. The FIFO can be accessed as a list 
 * normally. See list.h for more details on the interface.
 */


#include "dague_config.h"
#include "list.h"

/* do not want typedef here, to avoid type warnings */
#define dague_fifo_t dague_list_t

#define dague_fifo_construct(fifo) dague_list_construct(fifo)
#define dague_fifo_destruct(fifo) dague_list_destruct(fifo)

#define dague_fifo_is_empty(fifo) dague_list_is_empty(fifo)
#define dague_fifo_nolock_is_empty(fifo) dague_list_nolock_is_empty(fifo)
#define dague_ufifo_is_empty(fifo) dague_fifo_nolock_is_empty(fifo)

/* altough these are synonims to list actions, we use inline instead of 
 * #define, to have proper stack trace during debugging */
static inline void
dague_fifo_push(dague_fifo_t* list, dague_list_item_t* item) {
    dague_list_push_back(list, item); 
}
static inline void
dague_fifo_nolock_push(dague_fifo_t* list, dague_list_item_t* item) { 
    dague_list_nolock_push_back(list, item); 
}
#define dague_ufifo_push(list, item) dague_fifo_nolock_push(list, item)

static inline void
dague_fifo_chain(dague_fifo_t* list, dague_list_item_t* items) {
    dague_list_chain_back(list, items);
}
static inline void
dague_fifo_nolock_chain(dague_fifo_t* list, dague_list_item_t* items) { 
    dague_list_nolock_chain_back(list, items);
}
#define dague_ufifo_chain(list, items) dague_fifo_nolock_chain(list, items)

static inline dague_list_item_t*
dague_fifo_pop(dague_fifo_t* list) {
    return dague_list_pop_front(list); 
}
static inline dague_list_item_t*
dague_fifo_try_pop(dague_fifo_t* list) {
    return dague_list_try_pop_front(list);
}
#define dague_fifo_tpop(list) dague_fifo_try_pop(list)
static inline dague_list_item_t* 
dague_fifo_nolock_pop(dague_fifo_t* list) { 
    return dague_list_nolock_pop_front(list); 
}
#define dague_ufifo_pop(list) dague_fifo_nolock_pop(list)

#endif  /* FIFO_H_HAS_BEEN_INCLUDED */
