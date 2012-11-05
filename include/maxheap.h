/*
 * Copyright (c) 2009-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef MAXHEAP_H_HAS_BEEN_INCLUDED
#define MAXHEAP_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "dague_internal.h"

#include "debug.h"
#include <dague/sys/atomic.h>
#include "list_item.h"
#include <stdlib.h>

/**
 * The structure implemented here is not thread safe. All concurent
 * accesses should be protected by the upper level.
 */

/* main struct holding size info and ID */
typedef struct dague_heap {
    dague_list_item_t list_item; /* to be compatible with the lists */
    unsigned int size;           /* used only during building */
    unsigned int priority;
    dague_execution_context_t * top;
} dague_heap_t;

/*
  allocates an empty heap as a correctly doubly-linked singleton list
  with the lowest possible priority
 */
static inline dague_heap_t* heap_create(void)
{
    dague_heap_t* heap = calloc(sizeof(dague_heap_t), 1);
    /* Point back to the parent structure */
    heap->list_item.list_next = (dague_list_item_t*)heap;
    heap->list_item.list_prev = (dague_list_item_t*)heap;
    return heap;
}

static inline void heap_destroy(dague_heap_t** heap)
{
    assert((*heap)->top == NULL);
    free(*heap);
    (*heap) = NULL;
}

void heap_insert(dague_heap_t * heap, dague_execution_context_t * elem);
dague_execution_context_t* heap_split_and_steal(dague_heap_t ** heap_ptr, dague_heap_t ** new_heap_ptr);

/*
 * Insertion is O(lg n), as we know exactly how to get to the next insertion point,
 * and the tree is manually balanced.
 * Overall build is O(n lg n)
 *
 * Destroys elem->list_item next and prev.
 */
void heap_insert(dague_heap_t * heap, dague_execution_context_t * elem)
{
    assert(heap != NULL);
    assert(elem != NULL);
    heap->size++;
    elem->list_item.list_next = NULL;
    elem->list_item.list_prev = NULL;

    if (heap->size == 1) {
        heap->top = elem;
    } else {
        dague_execution_context_t * parent = heap->top;
        unsigned int bitmask = 1, size = heap->size;
        // prime the bitmask
        int level_counter = 0, parents_size = 0;
        while (bitmask <= size) {
            bitmask = bitmask << 1;
            level_counter++;
        }
        parents_size = level_counter;

        dague_execution_context_t ** parents = calloc(sizeof(dague_execution_context_t *), level_counter);
        // now the bitmask is two places farther than we want it, so back down
        bitmask = bitmask >> 2;

        parents[--level_counter] = heap->top;
        // now move through tree
        while (bitmask > 1) {
            parent = (dague_execution_context_t*)((bitmask & size) ? parent->list_item.list_next : parent->list_item.list_prev);
            parents[--level_counter] = parent; // save parent
            bitmask = bitmask >> 1;
        }
        if (bitmask & size)
            parent->list_item.list_next = (dague_list_item_t*)elem;
        else
            parent->list_item.list_prev = (dague_list_item_t*)elem;

        // now bubble up to preserve max heap org.
        while( (level_counter < parents_size) &&
               (parents[level_counter] != NULL) &&
               (elem->priority > parents[level_counter]->priority) ) {
            parent = parents[level_counter];
            DEBUG3(("MH:\tswapping parent %p and elem %p (priorities: %d and %d)\n", parent, elem, parent->priority, elem->priority));
            /* first, fix our grandparent, if necessary */
            if (level_counter + 1 < parents_size && parents[level_counter + 1] != NULL) {
                dague_execution_context_t * grandparent = parents[level_counter + 1];
                // i.e. our parent has a parent
                if (grandparent->list_item.list_prev /* left */ == (dague_list_item_t*)parent)
                    grandparent->list_item.list_prev = (dague_list_item_t*)elem;
                else /* our grandparent's right child is our parent*/
                    grandparent->list_item.list_next = (dague_list_item_t*)elem;
            }

            /* next, fix our parent */
            dague_list_item_t * parent_left  = (dague_list_item_t*)parent->list_item.list_prev;
            dague_list_item_t * parent_right = (dague_list_item_t*)parent->list_item.list_next;
            parent->list_item.list_prev = elem->list_item.list_prev;
            parent->list_item.list_next = elem->list_item.list_next;

            /* lastly, fix ourselves */
            if (parent_left == (dague_list_item_t*)elem) {
                /* we're our parent's left child */
                elem->list_item.list_prev = (dague_list_item_t*)parent;
                elem->list_item.list_next = (dague_list_item_t*)parent_right;
            } else {
                /* we're out parent's right child */
                elem->list_item.list_prev = (dague_list_item_t*)parent_left;
                elem->list_item.list_next = (dague_list_item_t*)parent;
            }

            if (parent == heap->top)
                    heap->top = elem;

            level_counter++;
        }
    }

    /* set priority to top priority */
    heap->priority = heap->top->priority;

#if defined(DAGUE_DEBUG)
    char tmp[MAX_TASK_STRLEN];
    DEBUG3(("MH:\tInserted exec C %s (%p) into maxheap %p of size %u\n",
            dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, elem), elem, heap, heap->size));
#endif  /* defined(DAGUE_DEBUG) */
}

/*
 * split-and-steal (remove) is O(1), although the preceding
 * list search is probably O(n), technically, since eventually we
 * end up with a list of n/2 trees with single nodes
 *
 * This function expects one valid heap (heap that has at least one element)
 * and another pointer to a NULL heap pointer.
 * If you pass a NULL heap, the function will simply return NULL.
 * This function WILL destroy your heap if it empties it.
 * It will also MODIFY your stack appropriately. If both of your heap pointers
 * are NULL after it returns, there was only one element in the heap you passed.
 * If only the new_heap pointer is NULL, then you still have one (and ONLY ONE)
 * valid heap.
 * If your valid heap had at least 3 nodes, then the heap will actually be split,
 * a new heap pointer created and put on your stack.
 * No matter what happens, an execution_context is returned unless the heap was NULL.
 */
dague_execution_context_t * heap_split_and_steal(dague_heap_t ** heap_ptr, dague_heap_t ** new_heap_ptr)
{
    // if tree is empty, return NULL
    // if tree has only one node (top), return new heap with single node
    //    moved into to_use slot
    // if tree has left child but not right child, put left child in new tree

    dague_heap_t * heap = *heap_ptr; // shortcut to doing a bunch of (*heap_ptr)s
    dague_execution_context_t * to_use = NULL;
    (*new_heap_ptr) = NULL; // this should already be NULL, but if it's not, we'll fix that.

    if (heap != NULL) {
        assert(heap->top != NULL); // this heap should have been destroyed
        to_use = heap->top; // this will always be what we return, even if it's NULL, if a valid heap was passed
        if (heap->top->list_item.list_prev == NULL) {
            /* no left child, so 'top' is the only node */
            DEBUG3(("MH:\tDestroying heap %p\n", heap->top, heap->top->list_item.list_next, heap));
            heap->top = NULL;
            heap_destroy(heap_ptr);
            assert(*heap_ptr == NULL);
        } else { /* does have left child */
            if (heap->top->list_item.list_next /* right */ == NULL) {
                /* but doesn't have right child, so still not splitting */
                heap->top = (dague_execution_context_t*)heap->top->list_item.list_prev; // left
                heap->priority = heap->top->priority;
                /* set up doubly-linked singleton list in here, as DEFAULT scenario */
                heap->list_item.list_prev = (dague_list_item_t*)*heap_ptr;
                heap->list_item.list_next = (dague_list_item_t*)*heap_ptr;
            } else { // heap has at least 3 nodes, so we should be actually splitting
                (*new_heap_ptr) = heap_create();
                (*new_heap_ptr)->top = (dague_execution_context_t*)heap->top->list_item.list_prev; // left
                (*new_heap_ptr)->priority = (*new_heap_ptr)->top->priority;
                heap->top = (dague_execution_context_t*)heap->top->list_item.list_next;
                heap->priority = heap->top->priority;

                /* set up doubly-linked two-element list in here, as DEFAULT scenario */
                heap->list_item.list_prev = (dague_list_item_t*)(*new_heap_ptr);
                heap->list_item.list_next = (dague_list_item_t*)(*new_heap_ptr);
                (*new_heap_ptr)->list_item.list_prev = (dague_list_item_t*)heap;
                (*new_heap_ptr)->list_item.list_next = (dague_list_item_t*)heap;
                DEBUG3(("MH:\tSplit heap %p into itself and heap %p\n", heap, *new_heap_ptr));
            }
        }
        to_use->list_item.list_next = (dague_list_item_t*)to_use; // safety's
        to_use->list_item.list_prev = (dague_list_item_t*)to_use; // sake
    }
#if defined(DAGUE_DEBUG)
    if (to_use != NULL) {
        char tmp[MAX_TASK_STRLEN];
        DEBUG3(("MH:\tStole exec C %s (%p) from heap %p\n", dague_snprintf_execution_context(tmp, MAX_TASK_STRLEN, to_use), to_use, heap));
    }
#endif  /* defined(DAGUE_DEBUG) */
    return to_use;
}

#endif
