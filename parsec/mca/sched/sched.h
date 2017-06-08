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
 * Intent provide modular schedulers that can install themselves in the
 * PaRSEC runtime system.
 *
 */

#ifndef MCA_SCHED_H
#define MCA_SCHED_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/execution_unit.h"

struct parsec_barrier_t;

BEGIN_C_DECLS

/**
 * Structures for sched components
 */

struct parsec_sched_base_component_2_0_0 {
    mca_base_component_2_0_0_t      base_version;
    mca_base_component_data_2_0_0_t base_data;
};

typedef struct parsec_sched_base_component_2_0_0 parsec_sched_base_component_2_0_0_t;
typedef struct parsec_sched_base_component_2_0_0 parsec_sched_base_component_t;

/**
 * Structure for sched modules
 */

/**
 * Global initialization for the scheduler module. This function is called once
 * per parsec_context_t and allows the scheduling module to setup any global data
 * it needs. This call will be followed by a call to init.
 */
typedef int  (*parsec_sched_base_module_install_fn_t)(parsec_context_t* master);

/**
 * This call must follow the global_init call. It is called once per execution
 * flow, allowing the scheduler to setup any type of local information each flow
 * needs. During this call the scheduler is expected to setup the structures
 * based on the locality information available on the parsec_context_t. The
 * barrier provided is global for all execution flows in a parsec_context_t and
 * it must be used with care.
 */
typedef int  (*parsec_sched_base_module_flow_init_fn_t)(parsec_execution_unit_t* eu_context,
                                                       struct parsec_barrier_t*);
/**
 * A double linked ring of ready tasks are available for the scheduler to insert
 * in it's management structures. The distance where the insertion will start is
 * indicated by the distance argument, the largest the value the further away the
 * tasks will start to be inserted. The available range for the distance is
 * scheduler dependent, but all scheduler must accept unreasonable values as indication
 * for the most distant scheduling level. This call is only called on the target
 * execution flow, based on the runtime decision.
 */
typedef int  (*parsec_sched_base_module_schedule_fn_t)
                 (parsec_execution_unit_t* eu_context,
                  parsec_execution_context_t* new_context,
                  int32_t distance);
/**
 * Select the best candidate to be executed next. Returns the distance where the
 * returned candidate has been found (greater means further away). The distance
 * is more than a hint, if ignored live locks can happen.
 */
typedef parsec_execution_context_t *(*parsec_sched_base_module_select_fn_t)
                 (parsec_execution_unit_t *eu_context,
                  int32_t* distance);

/**
 * Dump runtime statistics.
 */
typedef void (*parsec_sched_base_module_stats_fn_t)(parsec_execution_unit_t* eu_context);

/**
 * Finalization. This function is called once for each parsec_context_t, and upon
 * completion the schedulers are supposed to have released all internal
 * resources. Special attention should be taken for destroying the data used by
 * the execution flows as there is no special call for this. However, it should
 * be assumed that when remove is called all existing execution flows have been
 * torn down, and their data can be safely released.
 */
typedef void (*parsec_sched_base_module_remove_fn_t)(parsec_context_t* master);

struct parsec_sched_base_module_1_0_0_t {
    parsec_sched_base_module_install_fn_t     install;
    parsec_sched_base_module_flow_init_fn_t   flow_init;
    parsec_sched_base_module_schedule_fn_t    schedule;
    parsec_sched_base_module_select_fn_t      select;
    parsec_sched_base_module_stats_fn_t       display_stats;
    parsec_sched_base_module_remove_fn_t      remove;
};

typedef struct parsec_sched_base_module_1_0_0_t parsec_sched_base_module_1_0_0_t;
typedef struct parsec_sched_base_module_1_0_0_t parsec_sched_base_module_t;

typedef struct parsec_sched_module_s {
    const parsec_sched_base_component_t *component;
    parsec_sched_base_module_t           module;
} parsec_sched_module_t;

/**
 * Macro for use in components that are of type sched
 */
#define PARSEC_SCHED_BASE_VERSION_2_0_0 \
    MCA_BASE_VERSION_2_0_0, \
    "sched", 2, 0, 0

END_C_DECLS

#endif
