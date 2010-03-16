/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DPLASMA_FREE_LIST_H_HAS_BEEN_INCLUDED
#define DPLASMA_FREE_LIST_H_HAS_BEEN_INCLUDED

#include "dplasma_config.h"
#include "lifo.h"

typedef struct dplasma_freelist_t {
    dplasma_atomic_lifo_t lifo;
    size_t elem_size;
} dplasma_freelist_t;

typedef struct dplasma_freelist_item_t {
    union {
        dplasma_list_item_t                 item;
        dplasma_freelist_t*                 origin;
    } upstream;
} dplasma_freelist_item_t;

/**
 *
 */
int dplasma_freelist_init( dplasma_freelist_t* freelist, size_t elem_size );

/**
 *
 */
int dplasma_freelist_fini( dplasma_freelist_t* freelist );

/**
 *
 */
dplasma_list_item_t* dplasma_freelist_get(dplasma_freelist_t* freelist);

/**
 *
 */
int dplasma_freelist_release( dplasma_freelist_item_t* item );

#endif  /* DPLASMA_FREE_LIST_H_HAS_BEEN_INCLUDED */
