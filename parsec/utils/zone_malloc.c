/*
 * Copyright (c) 2012-2024 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      Stony Brook University.  All rights reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/utils/zone_malloc.h"
#include "parsec/utils/debug.h"
#include "parsec/constants.h"

#include "parsec/class/list.h"
#include "parsec/class/parsec_rbtree.h"

#include <stdio.h>

typedef struct zone_malloc_chunk_list_t {
    parsec_rbtree_node_t super;
    parsec_list_t list; /* list of free segments of specific size */
    int nb_units;       /* size of free segments */
} zone_malloc_chunk_list_t;

static inline void
zone_malloc_chunk_list_construct( zone_malloc_chunk_list_t* item )
{
    item->nb_units = 0;
    PARSEC_OBJ_CONSTRUCT(&item->list, parsec_list_t);
}

PARSEC_OBJ_CLASS_INSTANCE(zone_malloc_chunk_list_t, parsec_rbtree_node_t,
                          zone_malloc_chunk_list_construct, NULL);

static zone_malloc_chunk_list_t* allocate_chunk_list(zone_malloc_t *gdata, int nb_units) {
    zone_malloc_chunk_list_t* fl;
    /* add a new segment */
    if (!parsec_lifo_is_empty(&gdata->rbtree_free_list)) {
        fl = (zone_malloc_chunk_list_t*)parsec_lifo_pop(&gdata->rbtree_free_list);
    } else {
        /* allocate new */
        fl = PARSEC_OBJ_NEW(zone_malloc_chunk_list_t);
    }
    PARSEC_LIST_ITEM_SINGLETON(&fl->super.super);
    fl->nb_units = nb_units;
    return fl;
}

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
    parsec_atomic_lock_t temp_unlock = PARSEC_ATOMIC_UNLOCKED;

    gdata = (zone_malloc_t*)malloc( sizeof(zone_malloc_t) );
    gdata->base         = base_ptr;
    gdata->unit_size    = _unit_size;
    gdata->max_segment  = _max_segment;
    gdata->next_tid     = 0;
    gdata->segments     = (segment_t *)malloc(sizeof(segment_t) * _max_segment);
    parsec_atomic_lock_init(&gdata->lock);
    parsec_rbtree_init(&gdata->rbtree, offsetof(zone_malloc_chunk_list_t, nb_units));
    PARSEC_OBJ_CONSTRUCT(&gdata->rbtree_free_list, parsec_lifo_t);
    for(int i = 0; i < _max_segment; i++) {
        SEGMENT_AT_TID(gdata, i)->status = SEGMENT_UNDEFINED;
        PARSEC_OBJ_CONSTRUCT(&SEGMENT_AT_TID(gdata, i)->super, parsec_list_item_t);
    }
    head = SEGMENT_AT_TID(gdata, 0);
    head->status = SEGMENT_EMPTY;
    head->nb_units = _max_segment;
    head->nb_prev  = 1; /**< This is to force SEGMENT_OF_TID( 0 - prev ) to return NULL */

    zone_malloc_chunk_list_t *fl = PARSEC_OBJ_NEW(zone_malloc_chunk_list_t);
    fl->nb_units = head->nb_units;
    parsec_list_push_back(&fl->list, &head->super);
    parsec_rbtree_insert(&gdata->rbtree, &fl->super);

    return gdata;
}

void* zone_malloc_fini(zone_malloc_t** gdata)
{
    void* base_ptr = (*gdata)->base;

    free( (*gdata)->segments );

    (*gdata)->max_segment = 0;
    (*gdata)->unit_size = 0;
    (*gdata)->base = NULL;

    /* Release all chunk_list nodes still in the rbtree, including the
     * whole-pool node created in zone_malloc_init. */
    zone_malloc_chunk_list_t* fl;
    while ((*gdata)->rbtree.root != (*gdata)->rbtree.nil) {
        fl = (zone_malloc_chunk_list_t*)(*gdata)->rbtree.root;
        parsec_rbtree_remove(&(*gdata)->rbtree, (*gdata)->rbtree.root);
        PARSEC_OBJ_RELEASE(fl);
    }

    /* Release chunk_list nodes retired to the free list.  These are already
     * out of the rbtree, so no tree operation is needed. */
    while (!parsec_lifo_nolock_is_empty(&(*gdata)->rbtree_free_list)) {
        fl = (zone_malloc_chunk_list_t*)parsec_lifo_nolock_pop(&(*gdata)->rbtree_free_list);
        PARSEC_OBJ_RELEASE(fl);
    }

    parsec_rbtree_fini(&(*gdata)->rbtree);
    free(*gdata);
    *gdata = NULL;
    return base_ptr;
}

void *zone_malloc(zone_malloc_t *gdata, size_t size)
{
    segment_t *current_segment, *next_segment, *new_segment;
    int next_tid, current_tid, new_tid;
    int nb_units;
    zone_malloc_chunk_list_t* fl;

    nb_units = (size + gdata->unit_size - 1) / gdata->unit_size;

    if (nb_units == 0) {
        return NULL;
    }

    parsec_atomic_lock(&gdata->lock);
    /* try to find the smallest possible element, or one size larger */
    fl = (zone_malloc_chunk_list_t*) parsec_rbtree_find_or_larger(&gdata->rbtree, nb_units);

    if (NULL == fl) {
        /* no segment found */
        parsec_atomic_unlock(&gdata->lock);
        return NULL;
    }

    current_segment = (segment_t *)parsec_list_nolock_pop_front(&fl->list);
    assert(current_segment->nb_units >= nb_units);
    current_segment->status = SEGMENT_FULL;
    int fl_emptied = parsec_list_nolock_is_empty(&fl->list);
    current_tid = (current_segment - gdata->segments);
    if (current_segment->nb_units > nb_units) {
        /* segment must be split */
        next_tid = current_tid + current_segment->nb_units;

        /* get the following segment and update its back-pointer */
        next_segment = SEGMENT_AT_TID(gdata, next_tid);
        if( NULL != next_segment ) {
            next_segment->nb_prev -= nb_units;
        }

        new_tid = current_tid + nb_units;
        new_segment = SEGMENT_AT_TID(gdata, new_tid);
        new_segment->status = SEGMENT_EMPTY;
        new_segment->nb_prev  = nb_units;
        new_segment->nb_units = current_segment->nb_units - nb_units;

        if (fl_emptied) {
            /* The chunk-list for the old size is now empty.  Try to reuse it
             * for the remainder by updating its key in place; this avoids a
             * remove+insert pair when the new size fits in the same BST
             * position (e.g., sequential allocation into a single large chunk). */
            if (parsec_rbtree_update_node(&gdata->rbtree, &fl->super,
                                          new_segment->nb_units) == PARSEC_SUCCESS) {
                fl->nb_units = new_segment->nb_units;
                parsec_list_nolock_push_front(&fl->list, &new_segment->super);
            } else {
                /* A node for the remainder size already exists; retire fl. */
                parsec_rbtree_remove(&gdata->rbtree, &fl->super);
                PARSEC_LIST_ITEM_SINGLETON(&fl->super.super);
                parsec_lifo_nolock_push(&gdata->rbtree_free_list, &fl->super.super);
                fl = (zone_malloc_chunk_list_t*)parsec_rbtree_find(&gdata->rbtree,
                                                                    new_segment->nb_units);
                assert(fl != NULL);
                parsec_list_nolock_push_front(&fl->list, &new_segment->super);
            }
        } else {
            fl = (zone_malloc_chunk_list_t*)parsec_rbtree_find(&gdata->rbtree, new_segment->nb_units);
            if (fl == NULL) {
                fl = allocate_chunk_list(gdata, new_segment->nb_units);
                parsec_rbtree_insert(&gdata->rbtree, &fl->super);
            }
            parsec_list_nolock_push_front(&fl->list, &new_segment->super);
        }

        /* reduce size of current segment */
        current_segment->nb_units = nb_units;
    } else if (fl_emptied) {
        /* No split and the chunk-list is now empty: remove it. */
        parsec_rbtree_remove(&gdata->rbtree, &fl->super);
        PARSEC_LIST_ITEM_SINGLETON(&fl->super.super);
        parsec_lifo_nolock_push(&gdata->rbtree_free_list, &fl->super.super);
    }
    /* found segment of right size, done */
    parsec_atomic_unlock(&gdata->lock);
    return (void*)(gdata->base + (current_tid * gdata->unit_size));
}

void zone_free(zone_malloc_t *gdata, void *add)
{
    segment_t *current_segment, *next_segment, *prev_segment;
    int current_tid, next_tid, prev_tid;
    off_t offset;
    zone_malloc_chunk_list_t *fl, *reuse_fl = NULL;

    parsec_atomic_lock(&gdata->lock);
    offset = (char*)add -gdata->base;
    assert( (offset % gdata->unit_size) == 0);
    current_tid = offset / gdata->unit_size;
    current_segment = SEGMENT_AT_TID(gdata, current_tid);

    if( NULL == current_segment ) {
        zone_malloc_error("address to free not allocated\n");
        parsec_atomic_unlock(&gdata->lock);
        return;
    }

    if( SEGMENT_EMPTY == current_segment->status ) {
        zone_malloc_error("double free (or other buffer overflow) error in ZONE allocation");
        parsec_atomic_unlock(&gdata->lock);
        return;
    }

    /* check if we can merge segments */
    current_segment->status = SEGMENT_EMPTY;

    prev_tid = current_tid - current_segment->nb_prev;
    prev_segment = SEGMENT_AT_TID(gdata, prev_tid);

    next_tid = current_tid + current_segment->nb_units;
    next_segment = SEGMENT_AT_TID(gdata, next_tid);

    /* Pre-compute the fully merged size so we can update an empty fl in place,
     * avoiding a remove+insert pair in the common sequential-free pattern. */
    int merged_nb_units = current_segment->nb_units;
    if (NULL != prev_segment && prev_segment->status == SEGMENT_EMPTY)
        merged_nb_units += prev_segment->nb_units;
    if (NULL != next_segment && next_segment->status == SEGMENT_EMPTY)
        merged_nb_units += next_segment->nb_units;

    if (NULL != prev_segment && prev_segment->status == SEGMENT_EMPTY) {
        fl = (zone_malloc_chunk_list_t*)parsec_rbtree_find(&gdata->rbtree, prev_segment->nb_units);
        assert(fl != NULL);
        parsec_list_nolock_remove(&fl->list, &prev_segment->super);
        if (parsec_list_nolock_is_empty(&fl->list)) {
            /* fl is empty and still in the tree; update its key to merged_nb_units
             * so we can reuse it without a remove+insert. */
            if (parsec_rbtree_update_node(&gdata->rbtree, &fl->super, merged_nb_units) == PARSEC_SUCCESS) {
                fl->nb_units = merged_nb_units;
                reuse_fl = fl;
            } else {
                /* A node for merged_nb_units already exists; retire fl. */
                parsec_rbtree_remove(&gdata->rbtree, &fl->super);
                PARSEC_LIST_ITEM_SINGLETON(&fl->super.super);
                parsec_lifo_nolock_push(&gdata->rbtree_free_list, &fl->super.super);
            }
        }
        /* We can merge prev and current */
        if( NULL != next_segment ) {
            next_segment->nb_prev += prev_segment->nb_units;
        }
        prev_segment->nb_units += current_segment->nb_units;

        /* Pretend we are now our prev, so that we merge with next if needed */
        current_segment = prev_segment;
        current_tid     = prev_tid;
    }

    if (NULL != next_segment && next_segment->status == SEGMENT_EMPTY) {
        fl = (zone_malloc_chunk_list_t*)parsec_rbtree_find(&gdata->rbtree, next_segment->nb_units);
        assert(fl != NULL);
        parsec_list_nolock_remove(&fl->list, &next_segment->super);
        if (parsec_list_nolock_is_empty(&fl->list)) {
            if (reuse_fl == NULL) {
                /* No prev merge reuse yet; try to reuse this fl. */
                if (parsec_rbtree_update_node(&gdata->rbtree, &fl->super, merged_nb_units) == PARSEC_SUCCESS) {
                    fl->nb_units = merged_nb_units;
                    reuse_fl = fl;
                } else {
                    parsec_rbtree_remove(&gdata->rbtree, &fl->super);
                    PARSEC_LIST_ITEM_SINGLETON(&fl->super.super);
                    parsec_lifo_nolock_push(&gdata->rbtree_free_list, &fl->super.super);
                }
            } else {
                /* Already have reuse_fl; retire this one. */
                parsec_rbtree_remove(&gdata->rbtree, &fl->super);
                PARSEC_LIST_ITEM_SINGLETON(&fl->super.super);
                parsec_lifo_nolock_push(&gdata->rbtree_free_list, &fl->super.super);
            }
        }
        /* We can merge current and next */
        next_tid += next_segment->nb_units;
        current_segment->nb_units += next_segment->nb_units;
        next_segment = SEGMENT_AT_TID(gdata, next_tid);
        if( NULL != next_segment ) {
            next_segment->nb_prev = current_segment->nb_units;
        }
    }

    /* add the merged chunk into the RB tree */
    if (reuse_fl != NULL) {
        /* reuse_fl is already in the tree with key = merged_nb_units */
        parsec_list_nolock_push_front(&reuse_fl->list, &current_segment->super);
    } else {
        fl = (zone_malloc_chunk_list_t*)parsec_rbtree_find(&gdata->rbtree, current_segment->nb_units);
        if (fl == NULL) {
            /* no chunk list, create a new entry */
            fl = allocate_chunk_list(gdata, current_segment->nb_units);
            parsec_rbtree_insert(&gdata->rbtree, &fl->super);
        }
        assert(fl != NULL);
        parsec_list_nolock_push_front(&fl->list, &current_segment->super);
    }
    parsec_atomic_unlock(&gdata->lock);
}

size_t zone_in_use(zone_malloc_t *gdata)
{
    size_t ret = 0;
    segment_t *current_segment;
    int current_tid;
    parsec_atomic_lock(&gdata->lock);
    /* check segments */
    for(current_tid = 0;
        (current_segment = SEGMENT_AT_TID(gdata, current_tid)) != NULL;
        current_tid += current_segment->nb_units) {
        if( current_segment->status == SEGMENT_FULL ) {
            ret += gdata->unit_size * current_segment->nb_units;
        }
    }
    parsec_atomic_unlock(&gdata->lock);
    return ret;
}

typedef struct zone_malloc_rbtree_debug_t {
    int level, output_id;
    const char* prefix;
} zone_malloc_rbtree_debug_t;

static void zone_rbtree_cb(parsec_rbtree_node_t *node, void *data) {
    zone_malloc_chunk_list_t *fl = (zone_malloc_chunk_list_t*)node;
    zone_malloc_rbtree_debug_t* info = (zone_malloc_rbtree_debug_t*)data;
    int len = 0;
    PARSEC_LIST_NOLOCK_ITERATOR(&fl->list, iter, (void)iter; ++len; );
    parsec_debug_verbose(info->level, info->output_id, "%srbtree node: %d segments a %d units",
                         info->prefix, len, fl->nb_units);
}

size_t zone_debug(zone_malloc_t *gdata, int level, int output_id, const char *prefix)
{
    segment_t *current_segment;
    int current_tid;
    size_t ret = 0;

    parsec_atomic_lock(&gdata->lock);
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
    zone_malloc_rbtree_debug_t info = {level, output_id, prefix};
    parsec_rbtree_foreach(&gdata->rbtree, zone_rbtree_cb, &info);

    parsec_atomic_unlock(&gdata->lock);

    return ret;
}
