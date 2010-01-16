/*
 * Copyright (c) 2009-2010 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DPLASMA_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED
#define DPLASMA_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED

#include <stdint.h>
#include <pthread.h>
#include "dequeue.h"
#include "barrier.h"

typedef struct dplasma_context_t dplasma_context_t;

typedef struct dplasma_execution_unit_t {
    int32_t eu_id;
    pthread_t pthread_id;
    struct dplasma_eu_profiling_t* eu_profile;
#if defined(DPLASMA_USE_LIFO) || defined(DPLASMA_USE_GLOBAL_LIFO)
    dplasma_atomic_lifo_t* eu_task_queue;
#else
    dplasma_dequeue_t* eu_task_queue;
    void* placeholder;
#endif  /* DPLASMA_USE_LIFO */
    dplasma_context_t* master_context;
#if !defined(DPLASMA_USE_GLOBAL_LIFO) && defined(HAVE_HWLOC)
    int8_t*  eu_steal_from;
#endif  /* !defined(DPLASMA_USE_GLOBAL_LIFO) */

    char* remote_dep_fw_mask;
} dplasma_execution_unit_t;

struct dplasma_context_t {
    int16_t nb_cores;
    int16_t nb_nodes;
    int32_t __dplasma_internal_finalization_in_progress;
    size_t remote_dep_fw_mask_sizeof;
    dplasma_barrier_t  barrier;
    dplasma_execution_unit_t execution_units[1];
  
#if defined(DPLASMA_USE_LIFO) || defined(DPLASMA_USE_GLOBAL_LIFO)
    dplasma_atomic_lifo_t* fwd_IN_dep_queue;
    dplasma_atomic_lifo_t* fwd_OUT_dep_queue;
#else
    dplasma_dequeue_t* fwd_IN_dep_queue;
    dplasma_dequeue_t* fwd_OUT_dep_queue;
#endif /*DPLASMA_USE_LIFO */
};

#endif  /* DPLASMA_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED */
