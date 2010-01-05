/*
 * Copyright (c) 2009      The University of Tennessee and The University
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
#ifdef DPLASMA_USE_LIFO
    dplasma_atomic_lifo_t eu_task_queue;
#elif defined(DPLASMA_USE_GLOBAL_LIFO)
    /* Nothing in this case */
#else
    dplasma_dequeue_t eu_task_queue;
    void* placeholder;
#endif  /* DPLASMA_USE_LIFO */
    dplasma_context_t* master_context;
} dplasma_execution_unit_t;

struct dplasma_context_t {
    int32_t nb_cores;
    int32_t eu_waiting;
    dplasma_barrier_t  barrier;
    dplasma_execution_unit_t execution_units[1];
};

#endif  /* DPLASMA_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED */
