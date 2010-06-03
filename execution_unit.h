/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGuE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED
#define DAGuE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED

#include <stdint.h>
#include <pthread.h>
#include "dequeue.h"
#include "barrier.h"
#include "profiling.h"
#include "Buf2Cache/buf2cache.h"
#include "hbbuffer.h"

#define PLACEHOLDER_SIZE 2

typedef struct DAGuE_context_t DAGuE_context_t;

typedef struct DAGuE_execution_unit_t {
    int32_t eu_id;
    pthread_t pthread_id;
#if defined(DAGuE_PROFILING)
    DAGuE_thread_profiling_t* eu_profile;
#endif /* DAGuE_PROFILING */
#if defined(DAGuE_USE_LIFO) || defined(DAGuE_USE_GLOBAL_LIFO)
    DAGuE_atomic_lifo_t* eu_task_queue;
#elif defined(HAVE_HWLOC)
    DAGuE_hbbuffer_t   *eu_task_queue;
#else
    DAGuE_dequeue_t    *eu_task_queue;
#  if PLACEHOLDER_SIZE
    struct DAGuE_execution_context_t* placeholder[PLACEHOLDER_SIZE];
    int placeholder_pop;
    int placeholder_push;
#  endif  /* PLACEHOLDER_SIZE */
#endif  /* DAGuE_USE_LIFO */

    DAGuE_context_t* master_context;

#if defined(HAVE_HWLOC)
    DAGuE_hbbuffer_t    **eu_hierarch_queues; 
    uint32_t                eu_nb_hierarch_queues;
    DAGuE_dequeue_t      *eu_system_queue;
#  if defined(DAGuE_CACHE_AWARE)
    cache_t *closest_cache;
#  endif
#endif /* HAVE_HWLOC */

    uint32_t* remote_dep_fw_mask;
} DAGuE_execution_unit_t;

struct DAGuE_context_t {
    volatile int32_t __DAGuE_internal_finalization_in_progress;
    int32_t nb_cores;
    volatile int32_t __DAGuE_internal_finalization_counter;
    int32_t nb_nodes;
    volatile uint32_t taskstodo;
    int32_t my_rank;
    DAGuE_barrier_t  barrier;

    size_t remote_dep_fw_mask_sizeof;
#if defined(DAGuE_USE_LIFO) || defined(DAGuE_USE_GLOBAL_LIFO)
    DAGuE_atomic_lifo_t* fwd_IN_dep_queue;
    DAGuE_atomic_lifo_t* fwd_OUT_dep_queue;
#else
    DAGuE_dequeue_t* fwd_IN_dep_queue;
    DAGuE_dequeue_t* fwd_OUT_dep_queue;
#endif /*DAGuE_USE_LIFO */

    pthread_t* pthreads;

    /* This field should always be the last one in the structure. Even if the
     * declared number of execution units is 1 when we allocate the memory
     * we will allocate more (as many as we need), so everything after this
     * field might be overwritten.
     */
    DAGuE_execution_unit_t* execution_units[1];
};

#endif  /* DAGuE_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED */
