/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef HBBUFFER_H_HAS_BEEN_INCLUDED
#define HBBUFFER_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "dague_internal.h"
#include "dague/debug.h"
#include "dague/sys/atomic.h"
#include <stdlib.h>
#include "dague/class/lifo.h"
#include "dague/class/list.h"

typedef struct dague_hbbuffer_s dague_hbbuffer_t;

/**
 * Hierarchical Bounded Buffers:
 *
 *   bounded buffers with a parent storage, to store elements
 *   that will be ejected from the current buffer at push time.
 */

/**
 * parent push function: takes a pointer to the parent store object, and
 * a pointer to the element that is ejected out of this bounded buffer because
 * of a push. elt must be stored in the parent store (linked list, hbbuffer, or
 * dequeue, etc...) before the function returns
 */
typedef void (*dague_hbbuffer_parent_push_fct_t)(void *store, dague_list_item_t *elt);

struct dague_hbbuffer_s {
    size_t size;       /**< the size of the buffer, in number of void* */
    size_t ideal_fill; /**< hint on the number of elements that should be there to increase parallelism */
    unsigned int assoc_core_num; // only exists for scheduler instrumentation
    void    *parent_store; /**< pointer to this buffer parent store */
    /** function to push element to the parent store */
    dague_hbbuffer_parent_push_fct_t parent_push_fct;
    volatile dague_list_item_t *items[1]; /**< array of elements */
};

static inline dague_hbbuffer_t *dague_hbbuffer_new(size_t size,  size_t ideal_fill,
                                                   dague_hbbuffer_parent_push_fct_t parent_push_fct,
                                                   void *parent_store)
{
    /** Must use calloc to ensure that all ites are set to NULL */
    dague_hbbuffer_t *n = (dague_hbbuffer_t*)calloc(1, sizeof(dague_hbbuffer_t) + (size-1)*sizeof(dague_list_item_t*));
    n->size = size;
    n->ideal_fill = ideal_fill;
        /** n->nbelt = 0; <not needed because callc */
    n->parent_push_fct = parent_push_fct;
    n->parent_store = parent_store;
    DEBUG3("HBB:\tCreated a new hierarchical buffer of %d elements\n", (int)size);
    return n;
}

static inline void dague_hbbuffer_destruct(dague_hbbuffer_t *b)
{
    free(b);
}

static inline void dague_hbbuffer_push_all(dague_hbbuffer_t *b, dague_list_item_t *elt)
{
    dague_list_item_t *next = elt;
    int i = 0, nbelt = 0;

    while( NULL != elt ) {
        /* Assume that we're going to push elt.
         * Remove the first element from the list, keeping the rest of the list in next
         */
        next = dague_list_item_ring_chop(elt);
        DAGUE_LIST_ITEM_SINGLETON(elt);
        /* Try to find a room for elt */
        for(; (size_t)i < b->size; i++) {
            if( 0 == dague_atomic_cas(&b->items[i], (uintptr_t) NULL, (uintptr_t) elt) )
                continue;
            DEBUG3( "HBB:\tPush elem %p in local queue %p at position %d\n", elt, b, i );
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

    DEBUG3("HBB:\tpushed %d elements. %s\n", nbelt, NULL != elt ? "More to push, go to father" : "Everything pushed - done");

    if( NULL != elt ) {
        if( NULL != next ) {
            dague_list_item_ring_push(next, elt);
        }
        else {
            dague_list_item_singleton(elt);
        }
        b->parent_push_fct(b->parent_store, elt);
    }
}

static inline void dague_hbbuffer_push_all_by_priority(dague_hbbuffer_t *b, dague_list_item_t *list)
{
    int i = 0;
    dague_execution_context_t *candidate, *best_context;
    dague_list_item_t *topush;
    int best_index;
    dague_list_item_t *ejected = NULL;
#define CTX(to) ((dague_execution_context_t*)(to))

    /* Assume that we're going to push list.
     * Remove the first element from the list, keeping the rest of the list in topush
     * Don't move this line inside the loop: sometimes, multiple iterations of the loop with
     * the same element are necessary.
     */
    topush = list;
    list = dague_list_item_ring_chop(topush);
    DAGUE_LIST_ITEM_SINGLETON(topush);
    while(topush != NULL) {
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
            if( A_LOWER_PRIORITY_THAN_B(candidate, best_context, dague_execution_context_priority_comparator) ) {
                best_index = i;
                best_context = candidate;
            }
        }

        if( best_context == CTX(topush) )
            best_context = NULL;

        if( best_index > -1 ) {
            /* found a nice place, try to CAS */
            if( 1 == dague_atomic_cas( &b->items[best_index], (uintptr_t) best_context, (uintptr_t) topush ) ) {
                /* Woohoo ! Success... */
#if defined(DAGUE_DEBUG_VERBOSE)
                char tmp[MAX_TASK_STRLEN];
#endif
                DEBUG3("HBB:\tPushed task %s in buffer %p.\n",
                        dague_snprintf_execution_context( tmp,  MAX_TASK_STRLEN, CTX(topush) ), b);

                if( NULL != best_context ) {
                    /* best_context is the lowest priority element, and it was removed from the
                     * list, which is arguably full. Keep it in the ejected list, preserving
                     * the priority ordering (reverse priority)
                     * Hopefully, best_context is already a singleton, because it was pushed by
                     * the same function
                     */

                    DEBUG3("HBB:\tEjected task %s from buffer %p.\n",
                            dague_snprintf_execution_context( tmp, 128, best_context ), b);

                    /* "Push" ejected after best_context, then consider ejected as best_context, to preserve the
                     * ordering of priorities in ejected.
                     */
                    if( NULL != ejected ) {
                        dague_list_item_ring_merge( (dague_list_item_t*)best_context, ejected );
                    }
                    ejected = (dague_list_item_t*)best_context;
                }

                if( NULL == list )
                    break; /* We pushed everything */

                topush = list;
                list = dague_list_item_ring_chop(topush);
                DAGUE_LIST_ITEM_SINGLETON(topush);
            } /* else ... Somebody stole my spot... Try again with the same topush element */
        } else {
            /* topush has been singletoned after chop */
            if( NULL != ejected ) {
                dague_list_item_ring_merge( topush, ejected );
            }
            ejected = topush;

            /* Because list is in decreasing priority order, any new element
             * should not find a spot either.
             * TODO: ejected is ordered in decreasing priority; list is ordered
             *       by decreasing priority; a merge should be done.
             */
            if( NULL != list )
                dague_list_item_ring_merge( ejected, list );

            /* List is full. Go to parent */
            break;
        }
    } /* end while( topush != NULL ) */

    DEBUG3("HBB:\t  %s\n", NULL != ejected ? "More to push, go to father" : "Everything pushed - done");

    if( NULL != ejected ) {
#if defined(DAGUE_DEBUG_VERBOSE)
        dague_list_item_t *it;
        char tmp[MAX_TASK_STRLEN];

        DEBUG3("HBB:\t Elements that overflow and are given to the parent are:\n");
        it = ejected;
        do {
            DEBUG3("HBB:\tPush Parent %s\n",
                    dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, CTX(it)));
            it = DAGUE_LIST_ITEM_NEXT(it);
        } while(it != ejected);
#endif

        b->parent_push_fct(b->parent_store, ejected);
    }
#undef CTX
}

/* This code is unsafe, since another thread may be inserting new elements.
 * Use is_empty in safe-checking only
 */
static inline int dague_hbbuffer_is_empty(dague_hbbuffer_t *b)
{
    unsigned int i;
    for(i = 0; i < b->size; i++)
        if( NULL != b->items[i] )
            return 0;
    return 1;
}

/* local declaration of dague_heap struct for debug printing */
/* real definition in maxheap.h */
typedef struct dague_heap_hh {
    dague_list_item_t list_item;
    unsigned int size;
    unsigned int priority;
    dague_execution_context_t * top;
} dague_heap_h;
// TODO: this is a hack, but is necessary until someone decides to remove the incompatible print
// statement from pop_best.

static inline dague_list_item_t *dague_hbbuffer_pop_best(dague_hbbuffer_t *b,
                                                         off_t priority_offset)
{
    unsigned int idx;
    dague_list_item_t *best_elt = NULL;
    int best_idx = -1;
    dague_list_item_t *candidate;

    do {
        best_elt = NULL;
        best_idx = -1;

        for(idx = 0; idx < b->size; idx++) {
            if( NULL == (candidate = (dague_list_item_t *)b->items[idx]) )
                continue;

            if( (NULL == best_elt) || A_HIGHER_PRIORITY_THAN_B(candidate, best_elt, priority_offset) ) {
                best_elt  = candidate;
                best_idx  = idx;
            }
        }

        if( NULL == best_elt)
            break;

    } while( dague_atomic_cas( &b->items[best_idx], (uintptr_t) best_elt, (uintptr_t) NULL ) == 0 );


    /** Removes the element from the buffer. */
#if defined(DAGUE_DEBUG_VERBOSE)
    if( best_elt != NULL ) {
        char tmp[MAX_TASK_STRLEN];
        if (priority_offset == offsetof(dague_heap_h, priority)) {
                DEBUG3("HBB:\tFound best element %s in heap %p in local queue %p at position %d\n",
                        dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, (dague_execution_context_t*)((dague_heap_h*)best_elt)->top), best_elt,
                        b, best_idx);
        }
        // TODO these print statements are the reason for the dague_heap_h hack above.
        else {
            DEBUG3("HBB:\tFound best element %s in local queue %p at position %d\n",
                    dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, (dague_execution_context_t*)best_elt),
                    b, best_idx);
        }
    }
#endif

    return best_elt;
}

#endif /* HBBUFFER_H_HAS_BEEN_INCLUDED */
