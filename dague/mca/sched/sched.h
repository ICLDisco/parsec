/*
 * Copyright (c) 2013-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * @file
 *
 * Scheduler framework component interface.
 *
 * Intent
 *
 * provide modular schedulers that can install themselves in the
 * dague runtime system.
 *
 */

#ifndef MCA_SCHED_H
#define MCA_SCHED_H

#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/execution_unit.h"

struct dague_barrier_t;

BEGIN_C_DECLS

/**
 * Structures for sched components
 */

struct dague_sched_base_component_2_0_0 {
    mca_base_component_2_0_0_t      base_version;
    mca_base_component_data_2_0_0_t base_data;
};

typedef struct dague_sched_base_component_2_0_0 dague_sched_base_component_2_0_0_t;
typedef struct dague_sched_base_component_2_0_0 dague_sched_base_component_t;

/**
 * Structure for sched modules
 */

/**
 * Global initialization for the scheduler module. This function is called once
 * per dague_context_t and allows the scheduling module to setup any global data
 * it needs. This call will be followed by a call to init.
 */
typedef int  (*dague_sched_base_module_install_fn_t)(dague_context_t* master);

/**
 * This call must follow the global_init call. It is called once per execution
 * flow, allowing the scheduler to setup any type of local information each flow
 * needs. During this call the scheduler is expected to setup the structures
 * based on the locality information available on the dague_context_t. The
 * barrier provided is global for all execution flows in a dague_context_t and
 * it must be used with care.
 */
typedef int  (*dague_sched_base_module_flow_init_fn_t)(dague_execution_unit_t* eu_context,
                                                       struct dague_barrier_t*);
/**
 * A double linked ring of ready tasks are available for the scheduler to insert
 * in the local queues. This call is only called on the target execution flow,
 * based on the runtime decision.
 */
typedef int  (*dague_sched_base_module_schedule_fn_t)(dague_execution_unit_t* eu_context,
                                                      dague_execution_context_t* new_context);
/**
 * Select the best candidate to be executed next.
 */
typedef dague_execution_context_t *(*dague_sched_base_module_select_fn_t)(dague_execution_unit_t *eu_context);

/**
 * Dump runtime statistics.
 */
typedef void (*dague_sched_base_module_stats_fn_t)(dague_execution_unit_t* eu_context);

/**
 * Finalization. This function is called once for each dague_context_t, and upon
 * completion the schedulers are supposed to have released all internal
 * resources. Special attention should be taken for destroying the data used by
 * the execution flows as there is no special call for this. However, it should
 * be assumed that when remove is called all existing execution flows have been
 * torn down, and their data can be safely released.
 */
typedef void (*dague_sched_base_module_remove_fn_t)(dague_context_t* master);

struct dague_sched_base_module_1_0_0_t {
    dague_sched_base_module_install_fn_t     install;
    dague_sched_base_module_flow_init_fn_t   flow_init;
    dague_sched_base_module_schedule_fn_t    schedule;
    dague_sched_base_module_select_fn_t      select;
    dague_sched_base_module_stats_fn_t       display_stats;
    dague_sched_base_module_remove_fn_t      remove;
};

typedef struct dague_sched_base_module_1_0_0_t dague_sched_base_module_1_0_0_t;
typedef struct dague_sched_base_module_1_0_0_t dague_sched_base_module_t;

typedef struct dague_sched_module_s {
    const dague_sched_base_component_t *component;
    dague_sched_base_module_t           module;
} dague_sched_module_t;

/**
 * Macro for use in components that are of type sched
 */
#define DAGUE_SCHED_BASE_VERSION_2_0_0 \
    MCA_BASE_VERSION_2_0_0, \
    "sched", 2, 0, 0

END_C_DECLS

#endif
