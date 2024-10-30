/*
 * Copyright (c) 2012-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/utils/zone_malloc.h"
#include "parsec/utils/debug.h"

#include <stdio.h>

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

zone_malloc_t* zone_malloc_init(void* base_ptr, int _max_segment, size_t _unit_size)
{
    zone_malloc_t *gdata;
    segment_t *head;

    if( NULL == base_ptr ) {
        zone_malloc_error("Cannot manage an empty memory region\n");
        return NULL;
    }

    gdata = (zone_malloc_t*)malloc( sizeof(zone_malloc_t) );
    gdata->base               = base_ptr;
    gdata->unit_size          = _unit_size;
    gdata->max_segment        = _max_segment;
    gdata->lock               = PARSEC_ATOMIC_UNLOCKED;
    gdata->next_tid = 0;
    gdata->segments = (segment_t *)malloc(sizeof(segment_t) * _max_segment);
#if defined(PARSEC_DEBUG)
    for(int i = 0; i < _max_segment; i++) {
        SEGMENT_AT_TID(gdata, i)->status = SEGMENT_UNDEFINED;
    }
#endif /* defined(PARSEC_DEBUG) */
    head = SEGMENT_AT_TID(gdata, 0);
    head->status = SEGMENT_EMPTY;
    head->nb_units = _max_segment;
    head->nb_prev  = 1; /**< This is to force SEGMENT_OF_TID( 0 - prev ) to return NULL */

    return gdata;
}

void* zone_malloc_fini(zone_malloc_t** gdata)
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

void *zone_malloc(zone_malloc_t *gdata, size_t size)
{
    segment_t *current_segment, *next_segment, *new_segment;
    int next_tid, current_tid, new_tid;
    int cycled_through = 0, nb_units;

    parsec_atomic_lock(&gdata->lock);
    /* Let's start with the last remembered free slot */
    current_tid = gdata->next_tid;
    nb_units = (size + gdata->unit_size - 1) / gdata->unit_size;

    do {
        current_segment = SEGMENT_AT_TID(gdata, current_tid);
        if( NULL == current_segment ) {
            /* Maybe there is a free slot in the beginning. Let's cycle at least once before we bail out */
            if( 0 != cycled_through ) break;
            current_tid = 0;
            cycled_through = 1;
            current_segment = SEGMENT_AT_TID(gdata, current_tid);
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
            parsec_atomic_unlock(&gdata->lock);
            return (void *)(gdata->base + (current_tid * gdata->unit_size));
        }

        current_tid += current_segment->nb_units;
    } while( current_tid != gdata->next_tid );

    parsec_atomic_unlock(&gdata->lock);
    return NULL;
}

void zone_free(zone_malloc_t *gdata, void *add)
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

size_t zone_in_use(zone_malloc_t *gdata)
{
    size_t ret = 0;
    segment_t *current_segment;
    int current_tid;
    for(current_tid = 0;
        (current_segment = SEGMENT_AT_TID(gdata, current_tid)) != NULL;
        current_tid += current_segment->nb_units) {
        if( current_segment->status == SEGMENT_FULL ) {
            ret += gdata->unit_size * current_segment->nb_units;
        }
    }
    return ret;
}


size_t zone_debug(zone_malloc_t *gdata, int level, int output_id, const char *prefix)
{
    segment_t *current_segment;
    int current_tid;
    size_t ret = 0;

    for(current_tid = 0;
        (current_segment = SEGMENT_AT_TID(gdata, current_tid)) != NULL;
        current_tid += current_segment->nb_units) {
        if( current_segment->status == SEGMENT_EMPTY ) {
            ret += gdata->unit_size * current_segment->nb_units;
            if( NULL != prefix )
                parsec_debug_verbose(level, output_id, "%sfree: %d units (%d bytes) from %p to %p",
                                     prefix,
                                     current_segment->nb_units, gdata->unit_size*current_segment->nb_units,
                                     gdata->base + current_tid * gdata->unit_size,
                                     gdata->base + (current_tid+current_segment->nb_units) * gdata->unit_size - 1);
        } else {
            if( NULL != prefix )
                parsec_debug_verbose(level, output_id, "%sused: %d units (%d bytes) from %p to %p",
                                     prefix,
                                     current_segment->nb_units, gdata->unit_size*current_segment->nb_units,
                                     gdata->base + current_tid * gdata->unit_size,
                                     gdata->base + (current_tid+current_segment->nb_units) * gdata->unit_size - 1);
        }
    }

    return ret;
}
