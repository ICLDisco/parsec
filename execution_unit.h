/*
 * Copyright (c) 2009      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DPLASMA_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED
#define DPLASMA_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED

#include "stdint.h"
#include "lifo.h"

typedef struct dplasma_context_t dplasma_context_t;

typedef struct dplasma_execution_unit_t {
    int32_t eu_id;
    struct dplasma_eu_profiling_t* eu_profile;
    dplasma_atomic_lifo_t eu_task_queue;
    dplasma_context_t* master_context;
} dplasma_execution_unit_t;

struct dplasma_context_t {
    int32_t nb_cores;
    int32_t eu_waiting;
    dplasma_execution_unit_t execution_units[1];
};

#endif  /* DPLASMA_EXECUTION_UNIT_H_HAS_BEEN_INCLUDED */
