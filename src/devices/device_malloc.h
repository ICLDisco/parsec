/*
 * Copyright (c) 2012-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _ZONE_MALLOC_H_
#define _ZONE_MALLOC_H_

#include "dague_config.h"

#include <stdlib.h>
#include <assert.h>

#define ZONE_MALLOC_UNIT_SIZE (1024*1024)

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
static inline zone_malloc_t*
zone_malloc_init(void* base_ptr, int _max_segment, size_t _unit_size);

/**
 * Release all resources related to the memory zone, including the zone itself.
 */
static inline void*
zone_malloc_fini(zone_malloc_t** gdata);

/**
 * Allocate a memory area of length nb_units. In worst case the search is linear
 * with the number of existing allocations.
 */
static inline void *zone_malloc(zone_malloc_t *gdata, int nb_units);

/**
 * Release a specific memory zone. When possible this memory zone is
 * merged with similar memory zones surrounding its position.
 */
static inline void zone_free(zone_malloc_t *gdata, void *add);

static inline void zone_malloc_error(const char *msg)
{
    fprintf(stderr, "%s", msg);
}

static inline segment_t *SEGMENT_AT_TID(zone_malloc_t *gdata, int tid)
{
    if( tid < 0 )
        return NULL;
    if( tid >= gdata->max_segment )
        return NULL;
    return &gdata->segments[tid];
}

static inline int TID_OF_SEGMENT(zone_malloc_t *gdata, segment_t *seg)
{
    off_t diff = ((char*)seg) - ((char*)gdata->segments);
    assert( (diff % sizeof(segment_t)) == 0 );
    return diff / sizeof(segment_t);
}

static inline zone_malloc_t*
zone_malloc_init(void* base_ptr, int _max_segment, size_t _unit_size)
{
    zone_malloc_t *gdata;
    segment_t *head;
    int i;

    if( NULL == base_ptr ) {
        zone_malloc_error("Cannot manage an empty memory region\n");
        return NULL;
    }

    gdata = (zone_malloc_t*)malloc( sizeof(zone_malloc_t) );
    gdata->base               = base_ptr;
    gdata->unit_size          = _unit_size;
    gdata->max_segment        = _max_segment;

    gdata->next_tid = 0;
    gdata->segments = (segment_t *)malloc(sizeof(segment_t) * _max_segment);
#if defined(DAGUE_DEBUG)
    for(i = 0; i < _max_segment; i++) {
        SEGMENT_AT_TID(gdata, i)->status = SEGMENT_UNDEFINED;
    }
#endif /* defined(DAGUE_DEBUG) */
    head = SEGMENT_AT_TID(gdata, 0);
    head->status = SEGMENT_EMPTY;
    head->nb_units = _max_segment;
    head->nb_prev  = 1; /**< This is to force SEGMENT_OF_TID( 0 - prev ) to return NULL */

    return gdata;
}

static inline void*
zone_malloc_fini(zone_malloc_t** gdata)
{
    void* base_ptr = (*gdata)->base;

    free( (*gdata)->segments );

    (*gdata)->max_segment = 0;
    (*gdata)->unit_size = 0;
    (*gdata)->base = NULL;
    free(*gdata);
    *gdata = NULL;
    return base_ptr;
}

static inline void *zone_malloc(zone_malloc_t *gdata, int nb_units)
{
    segment_t *current_segment, *next_segment, *new_segment;
    int next_tid, current_tid, new_tid;
    int cycled_through = 0;

    /* Let's start with the last remembered free slot */
    current_tid = gdata->next_tid;

    do {
        current_segment = SEGMENT_AT_TID(gdata, current_tid);
        if( NULL == current_segment ) {
            /* Maybe there is a free slot in the beginning. Let's cycle at least once before we bail out */
            if( cycled_through == 0 ) {
                current_tid = 0;
                cycled_through = 1;
                current_segment = SEGMENT_AT_TID(gdata, current_tid);
            } else {
                return NULL;
            }
        }

        if( current_segment->status == SEGMENT_EMPTY && current_segment->nb_units >= nb_units ) {
            current_segment->status = SEGMENT_FULL;
            if( current_segment->nb_units > nb_units ) {
                next_tid = current_tid + current_segment->nb_units;
                next_segment = SEGMENT_AT_TID(gdata, next_tid);
                if( NULL != next_segment )
                    next_segment->nb_prev -= nb_units;

                new_tid = current_tid + nb_units;
                new_segment = SEGMENT_AT_TID(gdata, new_tid);
                new_segment->status = SEGMENT_EMPTY;
                new_segment->nb_prev  = nb_units;
                new_segment->nb_units = current_segment->nb_units - nb_units;

                /* new_tid is a free slot, remember for next malloc */
                gdata->next_tid = new_tid;

                current_segment->nb_units = nb_units;
            }
            return (void*)(gdata->base + (current_tid * gdata->unit_size));
        }

        current_tid += current_segment->nb_units;
    } while( current_tid != gdata->next_tid );

    return NULL;
}

static inline void zone_free(zone_malloc_t *gdata, void *add)
{
    segment_t *current_segment, *next_segment, *prev_segment;
    int current_tid, next_tid, prev_tid;
    off_t offset;

    offset = (char*)add -gdata->base;
    assert( (offset % gdata->unit_size) == 0);
    current_tid = offset / gdata->unit_size;
    current_segment = SEGMENT_AT_TID(gdata, current_tid);

    if( NULL == current_segment ) {
        zone_malloc_error("address to free not allocated\n");
        return;
    }

    if( SEGMENT_EMPTY == current_segment->status ) {
        zone_malloc_error("double free (or other buffer overflow) error in ZONE allocation");
        return;
    }

    current_segment->status = SEGMENT_EMPTY;

    prev_tid = current_tid - current_segment->nb_prev;
    prev_segment = SEGMENT_AT_TID(gdata, prev_tid);

    next_tid = current_tid + current_segment->nb_units;
    next_segment = SEGMENT_AT_TID(gdata, next_tid);

    if( NULL != prev_segment && prev_segment->status == SEGMENT_EMPTY ) {
        /* We can merge prev and current */
        if( NULL != next_segment ) {
            next_segment->nb_prev += prev_segment->nb_units;
        }
        prev_segment->nb_units += current_segment->nb_units;

        /* Pretend we are now our prev, so that we merge with next if needed */
        current_segment = prev_segment;
        current_tid     = prev_tid;
    }

    /* current_tid is a free slot, remember for next malloc */
    gdata->next_tid = current_tid;

    if( NULL != next_segment && next_segment->status == SEGMENT_EMPTY ) {
        /* We can merge current and next */
        next_tid += next_segment->nb_units;
        current_segment->nb_units += next_segment->nb_units;
        next_segment = SEGMENT_AT_TID(gdata, next_tid);
        if( NULL != next_segment ) {
            next_segment->nb_prev = current_segment->nb_units;
        }
    }
}

#endif /* _ZONE_MALLOC_H_ */
