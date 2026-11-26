/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2026      Stony Brook University. All rights reserved.
 */

#ifndef MAXHEAP_H_HAS_BEEN_INCLUDED
#define MAXHEAP_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_config.h"
#include "parsec/class/parsec_heap.h"
#include "parsec/runtime.h"
#include "parsec/mempool.h"  /* needs the types of runtime.h */

BEGIN_C_DECLS

/**
 * Wrapper around parsec_binheap_t that adds the list_item field (so the heap
 * can be stored in scheduler lists) and an explicit 'priority' field (the
 * max priority of any task in the heap, used by parsec_hbbuffer_pop_best
 * to pick the best heap to steal from without traversing the tree).
 *
 * Not thread-safe; all concurrent accesses must be protected by the caller.
 *
 * The one exception is the 'priority' field: parsec_hbbuffer_pop_best reads
 * it from every heap it finds in a buffer, before it owns any of them, so
 * the read may happen after another thread took the heap, emptied it and
 * destroyed it. The storage of a heap therefore comes from a memory pool and
 * goes back to it when the heap is destroyed: it remains readable (with a
 * stale priority) until the memory pool itself is destructed.
 */
typedef struct parsec_task_heap_s {
    parsec_list_item_t  list_item;  /**< for compatibility with scheduler lists */
    unsigned int        priority;   /**< max priority of any task in this heap */
    parsec_binheap_t    heap;       /**< pointer-based max-heap storage */
    parsec_thread_mempool_t *mempool_owner; /**< where the storage of this heap returns */
} parsec_task_heap_t;

/**
 * Take an empty heap from the memory pool of a thread, as a singleton list
 * item with zero priority. The memory pool holds elements of
 * sizeof(parsec_task_heap_t) that keep their owner in mempool_owner, and it
 * must outlive every buffer in which a heap created from it was stored.
 */
parsec_task_heap_t* heap_create(parsec_thread_mempool_t *mempool);

/**
 * Return an empty heap to the memory pool it was created from. Asserts that
 * the heap is empty.
 */
void heap_destroy(parsec_task_heap_t** heap);

/** Insert a task into the heap, updating the stored max priority. */
void heap_insert(parsec_task_heap_t *heap, parsec_task_t *elem);

/**
 * Remove the maximum-priority task from the heap and, if the heap has at
 * least 3 nodes, split it into two sub-heaps for work stealing.
 * On return, *heap_ptr and *new_heap_ptr are the two sub-heaps (either
 * may be NULL if the original heap had fewer than 3 nodes). The new sub-heap
 * is created from mempool.
 */
parsec_task_t* heap_split_and_steal(parsec_task_heap_t **heap_ptr,
                                     parsec_task_heap_t **new_heap_ptr,
                                     parsec_thread_mempool_t *mempool);

/**
 * Remove the maximum-priority task from the heap.
 * If the heap becomes empty it is destroyed and *heap_ptr is set to NULL.
 */
parsec_task_t* heap_remove(parsec_task_heap_t **heap_ptr);

END_C_DECLS

#endif  /* MAXHEAP_H_HAS_BEEN_INCLUDED */
