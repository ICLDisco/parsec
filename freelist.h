/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGuE_FREE_LIST_H_HAS_BEEN_INCLUDED
#define DAGuE_FREE_LIST_H_HAS_BEEN_INCLUDED

#include "DAGuE_config.h"
#include "lifo.h"

typedef struct DAGuE_freelist_t {
    DAGuE_atomic_lifo_t lifo;
    size_t elem_size; /** The size of the elements in the freelist */
    int ondemand;     /** If allocation on demand is allowed */
} DAGuE_freelist_t;

typedef struct DAGuE_freelist_item_t {
    union {
        DAGuE_list_item_t                 item;
        DAGuE_freelist_t*                 origin;
    } upstream;
} DAGuE_freelist_item_t;

/**
 *
 */
int DAGuE_freelist_init( DAGuE_freelist_t* freelist, size_t elem_size, int ondemand );

/**
 *
 */
int DAGuE_freelist_fini( DAGuE_freelist_t* freelist );

/**
 *
 */
DAGuE_list_item_t* DAGuE_freelist_get(DAGuE_freelist_t* freelist);

/**
 *
 */
int DAGuE_freelist_release( DAGuE_freelist_item_t* item );

#endif  /* DAGuE_FREE_LIST_H_HAS_BEEN_INCLUDED */
