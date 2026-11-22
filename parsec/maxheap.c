/*
 * Copyright (c) 2009-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2026      Stony Brook University. All rights reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"

#include "parsec/utils/debug.h"
#include "parsec/sys/atomic.h"
#include "parsec/maxheap.h"

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

/* list_prev = left child, list_next = right child (same as parsec_heap.c) */
#define HLEFT(item)  ((parsec_list_item_t *)(item)->list_prev)
#define HRIGHT(item) ((parsec_list_item_t *)(item)->list_next)

/* Highest set bit: used to compute sub-heap sizes in heap_split_and_steal().
 * Operates on size_t (not unsigned int) so it stays correct for heaps whose
 * size exceeds UINT_MAX. */
static inline size_t hiBit(size_t n)
{
    n |= (n >>  1);
    n |= (n >>  2);
    n |= (n >>  4);
    n |= (n >>  8);
    n |= (n >> 16);
#if SIZE_MAX > 0xFFFFFFFFu
    n |= (n >> 32);
#endif
    return n - (n >> 1);
}

parsec_task_heap_t* heap_create(parsec_thread_mempool_t *mempool)
{
    /* The storage is recycled, not zeroed: the memory pool only maintains the
     * list item and mempool_owner, every other field must be set here. */
    parsec_task_heap_t *h = (parsec_task_heap_t*)parsec_thread_mempool_allocate(mempool);
    h->list_item.list_next = (parsec_list_item_t*)h;
    h->list_item.list_prev = (parsec_list_item_t*)h;
    h->priority = 0;
    /* parsec_task_t has no spare field to stamp a FIFO sequence into, so
     * priority ties are broken arbitrarily here, as before this heap was
     * backed by the shared parsec_heap engine. */
    parsec_heap_init(&h->heap, offsetof(parsec_task_t, priority), PARSEC_HEAP_NO_SEQ);
    return h;
}

void heap_destroy(parsec_task_heap_t **heap)
{
    assert(parsec_heap_is_empty(&(*heap)->heap));
    parsec_heap_fini(&(*heap)->heap);
    /* Not free(): a thread scanning the buffer this heap was taken from may
     * still read its priority, see parsec_hbbuffer_pop_best(). */
    parsec_thread_mempool_free((*heap)->mempool_owner, *heap);
    *heap = NULL;
}

void heap_insert(parsec_task_heap_t *heap, parsec_task_t *elem)
{
    assert(heap != NULL);
    assert(elem != NULL);
    parsec_heap_push(&heap->heap, &elem->super);
    heap->priority = (unsigned int)COMPARISON_VAL(heap->heap.top, heap->heap.comp_offset);

#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                         "MH:\tInserted exec C %s (%p) into maxheap %p of size %zu",
                         parsec_task_snprintf(tmp, MAX_TASK_STRLEN, elem), elem,
                         heap, heap->heap.size);
#endif
}

parsec_task_t* heap_remove(parsec_task_heap_t **heap_ptr)
{
    parsec_task_heap_t *heap = *heap_ptr;
    if (NULL == heap) return NULL;

    parsec_list_item_t *item = parsec_heap_pop(&heap->heap);
    if (NULL == item) return NULL;

    parsec_task_t *task = (parsec_task_t*)item;

    if (parsec_heap_is_empty(&heap->heap)) {
        heap_destroy(heap_ptr);
    } else {
        heap->priority = (unsigned int)COMPARISON_VAL(heap->heap.top, heap->heap.comp_offset);
        /* Restore singleton list links so the wrapper can be re-inserted into a scheduler list */
        heap->list_item.list_prev = (parsec_list_item_t*)*heap_ptr;
        heap->list_item.list_next = (parsec_list_item_t*)*heap_ptr;
    }

    task->super.list_next = (parsec_list_item_t*)task;  /* safety */
    task->super.list_prev = (parsec_list_item_t*)task;

#if defined(PARSEC_DEBUG_NOISIER)
    if (task != NULL) {
        char tmp[MAX_TASK_STRLEN];
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                             "MH:\tStole exec C %s (%p) from heap %p",
                             parsec_task_snprintf(tmp, MAX_TASK_STRLEN, task), task, *heap_ptr);
    }
#endif
    return task;
}

parsec_task_t*
heap_split_and_steal(parsec_task_heap_t **heap_ptr,
                     parsec_task_heap_t **new_heap_ptr,
                     parsec_thread_mempool_t *mempool)
{
    parsec_task_heap_t *heap = *heap_ptr;
    *new_heap_ptr = NULL;
    if (NULL == heap) return NULL;

    parsec_binheap_t *h = &heap->heap;
    assert(h->top != NULL);

    parsec_task_t *to_use = (parsec_task_t*)h->top;

    if (NULL == HLEFT(h->top)) {
        /* Only root — no children */
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                             "MH:\tDestroying heap %p (single node)", heap);
        h->top = NULL;
        h->size = 0;
        heap_destroy(heap_ptr);
        goto prepare_for_return;
    }

    if (NULL == HRIGHT(h->top)) {
        /* Root has only a left child (size == 2) */
        assert(h->size == 2);
        parsec_list_item_t *left = HLEFT(h->top);
        h->top = left;
        h->size = 1;
        heap->priority = (unsigned int)COMPARISON_VAL(left, h->comp_offset);
        heap->list_item.list_prev = (parsec_list_item_t*)*heap_ptr;
        heap->list_item.list_next = (parsec_list_item_t*)*heap_ptr;
        goto prepare_for_return;
    }

    /* >= 3 nodes: split into left (new_heap) and right (heap) subtrees */
    {
        size_t size     = h->size;
        size_t highBit  = hiBit(size);
        size_t twoBit   = highBit >> 1;

        *new_heap_ptr = heap_create(mempool);
        (*new_heap_ptr)->heap.comp_offset = h->comp_offset;

        parsec_list_item_t *left_top  = HLEFT(h->top);
        parsec_list_item_t *right_top = HRIGHT(h->top);

        (*new_heap_ptr)->heap.top = left_top;
        (*new_heap_ptr)->priority = (unsigned int)COMPARISON_VAL(left_top, h->comp_offset);

        h->top = right_top;
        heap->priority = (unsigned int)COMPARISON_VAL(right_top, h->comp_offset);

        if (twoBit & size) { /* last node is in the right subtree */
            h->size = ~highBit & size;
            (*new_heap_ptr)->heap.size = size - h->size - 1;
        } else {             /* last node is in the left subtree */
            (*new_heap_ptr)->heap.size = (size & ~highBit) + twoBit;
            h->size = size - (*new_heap_ptr)->heap.size - 1;
        }

        /* Form a two-element ring so the caller can re-singleton each side */
        heap->list_item.list_prev = (parsec_list_item_t*)(*new_heap_ptr);
        heap->list_item.list_next = (parsec_list_item_t*)(*new_heap_ptr);
        (*new_heap_ptr)->list_item.list_prev = (parsec_list_item_t*)heap;
        (*new_heap_ptr)->list_item.list_next = (parsec_list_item_t*)heap;
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                             "MH:\tSplit heap %p into itself and heap %p", heap, *new_heap_ptr);
    }

  prepare_for_return:
    PARSEC_LIST_ITEM_SINGLETON(to_use);

#if defined(PARSEC_DEBUG_NOISIER)
    {
        char tmp[MAX_TASK_STRLEN];
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output,
                             "MH:\tStole exec C %s (%p) from heap %p",
                             parsec_task_snprintf(tmp, MAX_TASK_STRLEN, to_use), to_use, *heap_ptr);
    }
#endif
    return to_use;
}
