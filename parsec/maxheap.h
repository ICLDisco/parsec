/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef MAXHEAP_H_HAS_BEEN_INCLUDED
#define MAXHEAP_H_HAS_BEEN_INCLUDED

#include "parsec/class/list_item.h"

/**
 * The structure implemented here is not thread safe. All concurent
 * accesses should be protected by the upper level.
 */

/* main struct holding size info and ID */
typedef struct parsec_heap {
    parsec_list_item_t list_item; /* to be compatible with the lists */
    unsigned int size;
    unsigned int priority;
    parsec_execution_context_t* top;
} parsec_heap_t;

/*
 allocates an empty heap as a correctly doubly-linked singleton list
 with the lowest possible priority
 */
parsec_heap_t* heap_create(void);

void heap_destroy(parsec_heap_t** heap);

void heap_insert(parsec_heap_t * heap, parsec_execution_context_t * elem);

parsec_execution_context_t*
heap_split_and_steal(parsec_heap_t ** heap_ptr,
                     parsec_heap_t ** new_heap_ptr);
parsec_execution_context_t * heap_remove(parsec_heap_t ** heap_ptr);

#endif  /* MAXHEAP_H_HAS_BEEN_INCLUDED */
