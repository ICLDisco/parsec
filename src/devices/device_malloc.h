/*
 * Copyright (c) 2012      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef _GPU_MALLOC_H_
#define _GPU_MALLOC_H_

#include "dague_config.h"

#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#define GPU_MALLOC_UNIT_SIZE (1024*1024)

#define SEGMENT_EMPTY      1
#define SEGMENT_FULL       2
#define SEGMENT_UNDEFINED  3

typedef struct segment {
    int status;     /* True if this segment is full, false if it is free */
    int nb_units;   /* Number of units on this segment */
    int nb_prev;    /* Number of units on the segment before */
} segment_t;

typedef struct gpu_malloc_s {
    char      *base;                 /* Base pointer              */
    segment_t *segments;             /* Array of available segments */
    size_t     unit_size;            /* Basic Unit                */
    int        max_segment;          /* Maximum number of segment */
    int        next_tid;             /* Next TID to look at for a malloc */
} gpu_malloc_t;


static inline gpu_malloc_t *gpu_malloc_init(int max_segment, size_t unit_size);
static inline void  gpu_malloc_fini(gpu_malloc_t *gdata);
static inline void *gpu_malloc(gpu_malloc_t *gdata, int nb_units);
static inline void  gpu_free(  gpu_malloc_t *gdata, void *ptr);

static inline void gpu_malloc_error(const char *msg)
{
    /* if( gpu_malloc_cback != NULL ) */
    /*     gpu_malloc_cback(msg);*/
    fprintf(stderr, "%s", msg);
}

static inline segment_t *SEGMENT_AT_TID(gpu_malloc_t *gdata, int tid)
{
    if( tid < 0 )
        return NULL;
    if( tid >= gdata->max_segment )
        return NULL;
    return &gdata->segments[tid];
}

static inline int TID_OF_SEGMENT(gpu_malloc_t *gdata, segment_t *seg)
{
    off_t diff = ((char*)seg) - ((char*)gdata->segments);
    assert( (diff % sizeof(segment_t)) == 0 );
    return diff / sizeof(segment_t);
}

static inline gpu_malloc_t *gpu_malloc_init(int _max_segment, size_t _unit_size)
{
    gpu_malloc_t *gdata = (gpu_malloc_t*)malloc( sizeof(gpu_malloc_t) );
    void *ptr = NULL;
    segment_t *head;
    cudaError_t rc;
    int i;

    gdata->base               = NULL;
    gdata->unit_size          = _unit_size;
    gdata->max_segment        = _max_segment;

    rc = (cudaError_t)cudaMalloc( &ptr,
                                  (_max_segment * gdata->unit_size) );
    gdata->base = ptr;
    if( (cudaSuccess != rc) || (NULL == gdata->base) ) {
        gpu_malloc_error("unable to allocate backend memory\n");
        free(gdata);
        return NULL;
    }

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

static inline void gpu_malloc_fini(gpu_malloc_t *gdata)
{
    cudaError_t rc;

    free( gdata->segments );

    rc = (cudaError_t)cudaFree(gdata->base);
    if( cudaSuccess != rc ) {
        gpu_malloc_error("Failed to free the GPU backend memory.\n");
    }
    gdata->max_segment = 0;
    gdata->unit_size = 0;
    gdata->base = NULL;
}

static inline void *gpu_malloc(gpu_malloc_t *gdata, int nb_units)
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

static inline void gpu_free(gpu_malloc_t *gdata, void *add)
{
    segment_t *current_segment, *next_segment, *prev_segment;
    int current_tid, next_tid, prev_tid;
    off_t offset;

    offset = (char*)add -gdata->base;
    assert( (offset % gdata->unit_size) == 0);
    current_tid = offset / gdata->unit_size;
    current_segment = SEGMENT_AT_TID(gdata, current_tid);

    if( NULL == current_segment ) {
        gpu_malloc_error("address to free not allocated\n");
        return;
    }

    if( SEGMENT_EMPTY == current_segment->status ) {
        gpu_malloc_error("double free (or other buffer overflow) error in GPU allocation");
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

#endif /* _GPU_MALLOC_H_ */
