/*
 * Copyright (c) 2012-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _ZONE_MALLOC_H_
#define _ZONE_MALLOC_H_

#include "parsec_config.h"

#include <stdlib.h>
#include <assert.h>

#define SEGMENT_EMPTY      1
#define SEGMENT_FULL       2
#define SEGMENT_UNDEFINED  3

typedef struct segment {
    int status;     /* True if this segment is full, false if it is free */
    int32_t nb_units;   /* Number of units on this segment */
    int32_t nb_prev;    /* Number of units on the segment before */
} segment_t;

typedef struct zone_malloc_s {
    char      *base;                 /* Base pointer              */
    segment_t *segments;             /* Array of available segments */
    size_t     unit_size;            /* Basic Unit                */
    int        max_segment;          /* Maximum number of segment */
    int        next_tid;             /* Next TID to look at for a malloc */
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

#endif /* _ZONE_MALLOC_H_ */
