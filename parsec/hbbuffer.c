/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/utils/debug.h"
#include "parsec/sys/atomic.h"
#include "parsec/hbbuffer.h"
#include "parsec/maxheap.h"

#include <stdlib.h>

parsec_hbbuffer_t*
parsec_hbbuffer_new(size_t size,  size_t ideal_fill,
                    parsec_hbbuffer_parent_push_fct_t parent_push_fct,
                    void *parent_store)
{
    /** Must use calloc to ensure that all ites are set to NULL */
    parsec_hbbuffer_t *n = (parsec_hbbuffer_t*)calloc(1, sizeof(parsec_hbbuffer_t) + (size-1)*sizeof(parsec_list_item_t*));
    n->size = size;
    n->ideal_fill = ideal_fill;
        /** n->nbelt = 0; <not needed because callc */
    n->parent_push_fct = parent_push_fct;
    n->parent_store = parent_store;
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "HBB:\tCreated a new hierarchical buffer of %d elements", (int)size);
    return n;
}

void parsec_hbbuffer_destruct(parsec_hbbuffer_t *b)
{
    free(b);
}

void
parsec_hbbuffer_push_all(parsec_hbbuffer_t *b,
                         parsec_list_item_t *elt,
                         int32_t distance)
{
    parsec_list_item_t *next = elt;
    int i = 0, nbelt = 0;

    if( (0 != distance) && (NULL != b->parent_push_fct) )
        goto push_upstream;

    while( NULL != elt ) {
        /* Assume that we're going to push elt.
         * Remove the first element from the list, keeping the rest of the list in next
         */
        next = parsec_list_item_ring_chop(elt);
        PARSEC_LIST_ITEM_SINGLETON(elt);
        /* Try to find a room for elt */
        for(; (size_t)i < b->size; i++) {
            if( 0 == parsec_atomic_cas_ptr(&b->items[i], NULL, elt) )
                continue;
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,  "HBB:\tPush elem %p in local queue %p at position %d", elt, b, i );
            /* Found an empty space to push the first element. */
            nbelt++;
            break;
        }

        if( (size_t)i == b->size ) {
            /* It was impossible to push elt */
            break;
        }
        i++;  /* this position is already filled */
        elt = next;
    }

    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "HBB:\tpushed %d elements. %s",
                         nbelt, NULL != elt ? "More to push, go to father" : "Everything pushed - done");

    if( NULL == elt ) return;

    if( NULL != next ) {
        parsec_list_item_ring_push(next, elt);
    }

  push_upstream:
    assert(NULL != b->parent_push_fct);
    b->parent_push_fct(b->parent_store, elt, distance - 1);
}

void
parsec_hbbuffer_push_all_by_priority(parsec_hbbuffer_t *b,
                                     parsec_list_item_t *list,
                                     int32_t distance)
{
    int i = 0;
    parsec_task_t *candidate, *best_context;
    parsec_list_item_t *topush;
    int best_index;
    parsec_list_item_t *ejected = NULL;
#define CTX(to) ((parsec_task_t*)(to))

    if( (0 != distance) && (NULL != b->parent_push_fct) ) {
        ejected = list;
        goto push_upstream;
    }

    /* Assume that we're going to push list.
     * Remove the first element from the list, keeping the rest of the list in topush
     * Don't move this line inside the loop: sometimes, multiple iterations of the loop with
     * the same element are necessary.
     */
    assert(NULL != list);
    topush = list;
    list = parsec_list_item_ring_chop(topush);
    PARSEC_LIST_ITEM_SINGLETON(topush);
    while(1) {
        /* Iterate on the list, find best position */
        best_index = -1;
        /* We need to find something with a lower priority than topush anyway */
        best_context = CTX(topush);
        for(i = 0; (size_t)i < b->size; i++) {
            if( NULL == (candidate = CTX(b->items[i])) ) {
                best_index = i;
                best_context = CTX(topush);
                break;
            }

            /* This cannot segfault as long as the freelist holding candidate
             * is not emptied (which should not happen now). However, this solution
             * is subject ot a form of ABA that might lead it to expel an element
             * whose priority was higher. But this is considered as a case rare enough
             * to ignore for now.
             * Alternative is to lock the elements, which is not a good idea
             */
            if( A_LOWER_PRIORITY_THAN_B(candidate, best_context, parsec_execution_context_priority_comparator) ) {
                best_index = i;
                best_context = candidate;
            }
        }

        if( best_context == CTX(topush) )
            best_context = NULL;

        if( best_index > -1 ) {
            /* found a nice place, try to CAS */
            if( 1 == parsec_atomic_cas_ptr( &b->items[best_index], best_context, topush ) ) {
                /* Woohoo ! Success... */
#if defined(PARSEC_DEBUG_NOISIER)
                char tmp[MAX_TASK_STRLEN];
#endif
                PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "HBB:\tPushed task %s in buffer %p.",
                        parsec_task_snprintf( tmp,  MAX_TASK_STRLEN, CTX(topush) ), b);

                if( NULL != best_context ) {
                    /* best_context is the lowest priority element, and it was removed from the
                     * list, which is arguably full. Keep it in the ejected list, preserving
                     * the priority ordering (reverse priority)
                     * Hopefully, best_context is already a singleton, because it was pushed by
                     * the same function
                     */

                    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "HBB:\tEjected task %s from buffer %p.",
                            parsec_task_snprintf( tmp, 128, best_context ), b);

                    /* "Push" ejected after best_context, then consider ejected as best_context, to preserve the
                     * ordering of priorities in ejected.
                     */
                    if( NULL != ejected ) {
                        parsec_list_item_ring_merge( (parsec_list_item_t*)best_context, ejected );
                    }
                    ejected = (parsec_list_item_t*)best_context;
                }

                if( NULL == list )
                    break; /* We pushed everything */

                topush = list;
                list = parsec_list_item_ring_chop(topush);
                PARSEC_LIST_ITEM_SINGLETON(topush);
            } /* else ... Somebody stole my spot... Try again with the same topush element */
        } else {
            /* topush has been singletoned after chop */
            if( NULL != ejected ) {
                parsec_list_item_ring_merge( topush, ejected );
            }
            ejected = topush;

            /* Because list is in decreasing priority order, any new element
             * should not find a spot either.
             * TODO: ejected is ordered in decreasing priority; list is ordered
             *       by decreasing priority; a merge should be done.
             */
            if( NULL != list )
                parsec_list_item_ring_merge( ejected, list );

            /* List is full. Go to parent */
            break;
        }
    }

  push_upstream:
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "HBB:\t  %s",
                         NULL != ejected ? "More to push, go to father" : "Everything pushed - done");

    if( NULL != ejected ) {
#if defined(PARSEC_DEBUG_NOISIER)
        parsec_list_item_t *it;
        char tmp[MAX_TASK_STRLEN];

        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "HBB:\t Elements that overflow and are given to the parent are:");
        it = ejected;
        do {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "HBB:\tPush Parent %s",
                    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, CTX(it)));
            it = PARSEC_LIST_ITEM_NEXT(it);
        } while(it != ejected);
#endif
        assert(NULL != b->parent_push_fct);
        b->parent_push_fct(b->parent_store, ejected, distance - 1);
    }
#undef CTX
}

parsec_list_item_t*
parsec_hbbuffer_pop_best(parsec_hbbuffer_t *b, off_t priority_offset)
{
    unsigned int idx;
    parsec_list_item_t *best_elt = NULL;
    int best_idx = -1;
    parsec_list_item_t *candidate;

    do {
        best_elt = NULL;
        best_idx = -1;

        for(idx = 0; idx < b->size; idx++) {
            if( NULL == (candidate = (parsec_list_item_t *)b->items[idx]) )
                continue;

            if( (NULL == best_elt) || A_HIGHER_PRIORITY_THAN_B(candidate, best_elt, priority_offset) ) {
                best_elt  = candidate;
                best_idx  = idx;
            }
        }

        if( NULL == best_elt)
            break;

    } while( parsec_atomic_cas_ptr( &b->items[best_idx], best_elt, NULL ) == 0 );

    /** Removes the element from the buffer. */
#if defined(PARSEC_DEBUG_NOISIER)
    if( best_elt != NULL ) {
        char tmp[MAX_TASK_STRLEN];
        if (priority_offset == offsetof(parsec_heap_t, priority)) {
                PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "HBB:\tFound best element %s in heap %p in local queue %p at position %d",
                        parsec_task_snprintf(tmp, MAX_TASK_STRLEN, (parsec_task_t*)((parsec_heap_t*)best_elt)->top), best_elt,
                        b, best_idx);
        } else {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "HBB:\tFound best element %s in local queue %p at position %d",
                    parsec_task_snprintf(tmp, MAX_TASK_STRLEN, (parsec_task_t*)best_elt),
                    b, best_idx);
        }
    }
#endif

    return best_elt;
}
