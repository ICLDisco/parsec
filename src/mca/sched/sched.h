/*
 * Copyright (c) 2013      The University of Tennessee and The University
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
#include "execution_unit.h"

BEGIN_C_DECLS

/**
 * Structures for sched components
 */

struct dague_sched_base_component_2_0_0 {
    mca_base_component_2_0_0_t base_version;
    mca_base_component_data_2_0_0_t base_data;
};

typedef struct dague_sched_base_component_2_0_0 dague_sched_base_component_2_0_0_t;
typedef struct dague_sched_base_component_2_0_0 dague_sched_base_component_t;


/**
 * Structure for sched modules
 */

typedef int  (*dague_sched_base_module_install_fn_t)(dague_context_t* master);
typedef int  (*dague_sched_base_module_schedule_fn_t)(dague_execution_unit_t* eu_context, dague_execution_context_t* new_context);
typedef dague_execution_context_t *(*dague_sched_base_module_select_fn_t)( dague_execution_unit_t *eu_context );
typedef void (*dague_sched_base_module_stats_fn_t)(dague_execution_unit_t* eu_context);
typedef void (*dague_sched_base_module_remove_fn_t)(dague_context_t* master);

struct dague_sched_base_module_1_0_0_t {
    dague_sched_base_module_install_fn_t    install;
    dague_sched_base_module_schedule_fn_t   schedule;
    dague_sched_base_module_select_fn_t     select;
    dague_sched_base_module_stats_fn_t      display_stats;
    dague_sched_base_module_remove_fn_t     remove;
};

typedef struct dague_sched_base_module_1_0_0_t dague_sched_base_module_1_0_0_t;
typedef struct dague_sched_base_module_1_0_0_t dague_sched_base_module_t;

typedef struct {
    const dague_sched_base_component_t *component;
    dague_sched_base_module_t     module;
} dague_sched_module_t;

/**
 * Macro for use in components that are of type sched
 */
#define DAGUE_SCHED_BASE_VERSION_2_0_0 \
    MCA_BASE_VERSION_2_0_0, \
    "sched", 2, 0, 0

END_C_DECLS

#endif
