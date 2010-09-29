/*
 * Copyright (c) 2010      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef FIFO_H_HAS_BEEN_INCLUDED
#define FIFO_H_HAS_BEEN_INCLUDED

#include "dague_config.h"

typedef struct dague_fifo_t {
    dague_list_item_t  fifo_ghost;
} dague_fifo_t;

static inline int dague_fifo_is_empty( dague_fifo_t* fifo )
{
    return (fifo->fifo_ghost.list_next == &(fifo->fifo_ghost) ? 1 : 0);
}

static inline dague_list_item_t* dague_fifo_push( dague_fifo_t* fifo,
                                                  dague_list_item_t* elem )
{
    elem->list_prev = (dague_list_item_t *)fifo->fifo_ghost.list_prev;
    elem->list_next = &(fifo->fifo_ghost);
    elem->list_prev->list_next = elem;
    elem->list_next->list_prev = elem;
    return elem;
}

static inline dague_list_item_t* dague_fifo_pop( dague_fifo_t* fifo )
{
    dague_list_item_t* elem = (dague_list_item_t *)fifo->fifo_ghost.list_next;
    elem->list_next->list_prev = elem->list_prev;
    elem->list_prev->list_next = elem->list_next;
    return (elem == &(fifo->fifo_ghost) ? NULL : elem);
}

static inline void dague_fifo_construct( dague_fifo_t* fifo )
{
    fifo->fifo_ghost.list_next = &(fifo->fifo_ghost);
    fifo->fifo_ghost.list_prev = &(fifo->fifo_ghost);
}

#endif  /* FIFO_H_HAS_BEEN_INCLUDED */

