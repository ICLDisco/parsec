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

typedef struct segment {
    int start_index;/* Index of the first byte of this segment */
    int nb_units;   /* Number of units occupied by this segment */
    int nb_free;    /* Number of units free after this segment */
    struct segment *next;
} segment_t;

typedef struct gpu_malloc_s {
    char      *base;                 /* Base pointer              */
    segment_t *allocated_segments;   /* List of allocated segment */
    segment_t *free_segments;        /* List of available segment */
    size_t     unit_size;            /* Nasic Unit                */
    int        max_segment;          /* Maximum number of segment */
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

static inline gpu_malloc_t *gpu_malloc_init(int _max_segment, size_t _unit_size)
{
    gpu_malloc_t *gdata = (gpu_malloc_t*)malloc( sizeof(gpu_malloc_t) );
    segment_t *s;
    cudaError_t rc;
    int i;

    gdata->base               = NULL;
    gdata->allocated_segments = NULL;
    gdata->free_segments      = NULL;
    gdata->unit_size          = _unit_size;
    gdata->max_segment        = _max_segment+2;

    rc = (cudaError_t)cudaMalloc( (void**)(&(gdata->base)),
                                  (_max_segment * gdata->unit_size) );
    if( (cudaSuccess != rc) || (NULL == gdata->base) ) {
        gpu_malloc_error("unable to allocate backend memory\n");
        free(gdata);
        return NULL;
    }

    for(i = 0 ; i < _max_segment; i++) {
        s = (segment_t*)malloc(sizeof(segment_t));
        s->next = gdata->free_segments;
        gdata->free_segments = s;
    }

    /* First and last segments are persistent. Simplifies the algorithm */
    gdata->allocated_segments = (segment_t*)malloc(sizeof(segment_t));
    gdata->allocated_segments->start_index = 0;
    gdata->allocated_segments->nb_units    = 1;
    gdata->allocated_segments->nb_free     = _max_segment;

    gdata->allocated_segments->next = (segment_t*)malloc(sizeof(segment_t));
    gdata->allocated_segments->next->start_index = _max_segment+1;
    gdata->allocated_segments->next->nb_units    = 1;
    gdata->allocated_segments->next->nb_free     = 0;
    gdata->allocated_segments->next->next        = NULL;

    return gdata;
}

static inline void gpu_malloc_fini(gpu_malloc_t *gdata)
{
    segment_t *s;
    cudaError_t rc;

    while( NULL != gdata->allocated_segments ) {
        s = gdata->allocated_segments->next;
        free(gdata->allocated_segments);
        gdata->allocated_segments = s;
    }

    while( NULL != gdata->free_segments ) {
        s = gdata->free_segments->next;
        free(gdata->free_segments);
        gdata->free_segments = s;
    }

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
    segment_t *s, *n;

    for(s = gdata->allocated_segments; s->next != NULL; s = s->next) {
        if ( s->nb_free > nb_units ) {
            assert(nb_units > 0);

            n = gdata->free_segments;
            gdata->free_segments = gdata->free_segments->next;

            n->start_index = s->start_index + s->nb_units;
            n->nb_units = nb_units;
            n->nb_free = s->nb_free - n->nb_units;
            n->next = s->next;
            s->nb_free = 0;
            s->next = n;
            return (void*)(gdata->base + (n->start_index * gdata->unit_size));
        }
    }

    return NULL;
}

static inline void gpu_free(gpu_malloc_t *gdata, void *add)
{
    segment_t *s, *p;
    int tid;

    p   = gdata->allocated_segments;
    tid = ((char*)add - gdata->base) / gdata->unit_size;

    for(s = gdata->allocated_segments->next; s->next != NULL; s = s->next) {
        if ( s->start_index == tid ) {
            p->next = s->next;
            p->nb_free += s->nb_units + s->nb_free;

            s->next = gdata->free_segments;
            gdata->free_segments = s;

            return;
        }
        p = s;
    }
    gpu_malloc_error("address to free not allocated\n");
}

#endif /* _GPU_MALLOC_H_ */
