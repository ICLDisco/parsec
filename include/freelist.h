/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_FREE_LIST_H_HAS_BEEN_INCLUDED
#define DAGUE_FREE_LIST_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "lifo.h"

typedef struct dague_freelist_t {
    dague_atomic_lifo_t lifo;
    size_t elem_size; /** The size of the elements in the freelist */
    int ondemand;     /** If allocation on demand is allowed */
} dague_freelist_t;

typedef struct dague_freelist_item_t {
    union {
        dague_list_item_t                 item;
        dague_freelist_t*                 origin;
    } upstream;
} dague_freelist_item_t;

/**
 *
 */
int dague_freelist_init( dague_freelist_t* freelist, size_t elem_size, int ondemand );

/**
 *
 */
int dague_freelist_fini( dague_freelist_t* freelist );

/**
 *
 */
dague_list_item_t* dague_freelist_get(dague_freelist_t* freelist);

/**
 *
 */
int dague_freelist_release( dague_freelist_item_t* item );

#endif  /* DAGUE_FREE_LIST_H_HAS_BEEN_INCLUDED */
