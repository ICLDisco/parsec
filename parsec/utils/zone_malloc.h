/*
 * Copyright (c) 2012-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _ZONE_MALLOC_H_
#define _ZONE_MALLOC_H_

#include "parsec/parsec_config.h"
#include "parsec/sys/atomic.h"

#include "parsec/class/parsec_rbtree.h"
#include "parsec/class/lifo.h"

#include <stdlib.h>
#include <assert.h>

BEGIN_C_DECLS

#define SEGMENT_EMPTY      1
#define SEGMENT_FULL       2
#define SEGMENT_UNDEFINED  3

typedef struct segment {
    parsec_list_item_t super;
    int status;     /* True if this segment is full, false if it is free */
    int nb_units;   /* Number of units on this segment */
    int nb_prev;    /* Number of units on the segment before */
} segment_t;

typedef struct zone_malloc_s {
    char      *base;                 /* Base pointer              */
    segment_t *segments;             /* Array of segments */
    size_t     unit_size;            /* Basic Unit                */
    int        max_segment;          /* Maximum number of segment */
    int        next_tid;             /* Next TID to look at for a malloc */
    parsec_atomic_lock_t lock;
    parsec_rbtree_t rbtree;          /* RB tree tracking chunks of free segments */
    parsec_lifo_t rbtree_free_list;
} zone_malloc_t;


/**
 * Define a memory allocator starting from base_ptr, with a length of
 * _max_segment * _unit_size bytes. The base_ptr can be any type of
 * memory, it is not directly used by the allocator.
 */
zone_malloc_t* zone_malloc_init(void* base_ptr, int _max_segment, size_t _unit_size);

/**
 * Release all resources related to the memory zone, including the zone itself.
 */
void* zone_malloc_fini(zone_malloc_t** gdata);

/**
 * Allocate a memory area of length size bytes. In worst case the search is linear
 * with the number of existing allocations.
 */
void *zone_malloc(zone_malloc_t *gdata, size_t size);

/**
 * Release a specific memory zone. When possible this memory zone is
 * merged with similar memory zones surrounding its position.
 */
void zone_free(zone_malloc_t *gdata, void *add);

/**
 * Computes how much memory is in use
 */
size_t zone_in_use(zone_malloc_t *gdata);

/**
 * Prints information on the amount of available blocks
 * Do not print anything if prefix is NULL
 */
size_t zone_debug(zone_malloc_t *gdata, int level, int output_id, const char *prefix);

END_C_DECLS

#endif /* _ZONE_MALLOC_H_ */
