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

/** @addtogroup parsec_internal_scheduling
 *  @{
 *
 * @file
 *
 * Scheduler framework component interface.
 *
 * This Modular Component provides an End-User API to 
 * implement a PaRSEC runtime scheduler.
 *
 * @section SchedGeneralPrinciple General Principle
 *
 * The scheduler is fully distributed: each execution stream of a same
 * process alternate between executing a task and scheduling tasks. Once
 * a task is executed, the task engine computes a set of tasks that become
 * ready. The main role of the scheduler is two-fold:
 *   - when new tasks become ready, it must dispatch this set of tasks 
 *     into scheduler-specific data structures for later selection
 *     (this is the schedule step)
 *   - when requested, it must select a task for execution (this is
 *     the selection step).
 * To manage the set of ready tasks, each execution stream has access to
 * an opaque pointer in each execution_unit_t structure that they can
 * use to point to lists, queues, or arrays of tasks, as they seem fit.
 *
 * @section SchedInstallation Installation / Removal
 *
 * The scheduler is selected dynamically using the Modular Component Architecture
 * mechanism. It can change at runtime, between two execution activities
 * (e.g. once a parsec_context_t has been waited upon, and before any new
 * parsec_handle_t has been enqueued / started in that parsec_context_t).
 * 
 * When a scheduler is changed, the previous one is removed, and the new
 * one is installed. Initially, a first one is installed.
 *
 * Installation is a two-steps process: first a global installation
 * function is called once on the parsec_context_t. Then, each execution
 * stream calls a flow_init function on its own parsec_execution_unit_t.
 * The first function should be used only to define global parameters.
 * The second function should setup the data structures used by the
 * scheduler, and point to them through the scheduler_object opaque
 * pointer that is available for this use in each parsec_execution_unit_t.
 * To help with the synchronization and sharing of structures, a
 * parsec_barrier_t is passed to the flow_init function, and all
 * execution streams call the flow_init function together.
 *
 * @section SchedFairness Fairness and Distance
 *
 * Some tasks may decide to return before their completion and request
 * to be re-scheduled later. Their progress may (sometimes) depend upon
 * the execution of other actions that may happen in other tasks. As a
 * consequence, the schedulers must be fair and cannot select the same
 * task until it is executed, if that task keep asking to be delayed.
 *
 * To ensure fairness of execution, PaRSEC relies on a distance concept.
 *
 * Seeing the set of ready tasks as a (virtually ordered) list, a task
 * that is scheduled with a small distance can be early in the list, 
 * while a task that is scheduled with a large distance should be found
 * later in the list.
 *
 * The selection operation allows the user-defined scheduler to return
 * an integer, the distance, that represent how far away from the head
 * of the virtually ordered list the task was selected.
 *
 * The schedule operation provides a mandatory hint, the distance, that
 * says how far away from the head of the virtually ordered list of tasks
 * this set of tasks should be pushed.
 *
 * This concept is a hint: it needs not be followed or computed exactly;
 * However this hint is mandatory: a fair scheduler cannot repetedly select
 * a task that was scheduled at distance X if there are still tasks scheduled
 * at distance Y<X. A task that cannot execute now will kept being scheduled
 * with a distance larger than its selection distance.
 */

#ifndef MCA_SCHED_H
#define MCA_SCHED_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/execution_unit.h"

struct parsec_barrier_t;

BEGIN_C_DECLS

struct parsec_sched_base_component_2_0_0 {
    mca_base_component_2_0_0_t      base_version;
    mca_base_component_data_2_0_0_t base_data;
};

typedef struct parsec_sched_base_component_2_0_0 parsec_sched_base_component_2_0_0_t;
typedef struct parsec_sched_base_component_2_0_0 parsec_sched_base_component_t;

/**
 * @brief Global Initialization for the scheduler module.
 *
 * @details 
 * This function is called once per parsec_context_t and
 * allows the scheduling module to setup any global data it
 * needs. This call will be followed by a call to init.  
 * @param[inout] master a pointer to the PaRSEC context
 * @code{c}
 * For each vp in [0; master->nb_vp];
 *     For each eu in [0; master->virtual_process[vp]->nb_vp];
 *         master->virtual_process[vp]->execution_units[eu]->scheduler_object
 * @endcode
 * is a pointer to an opaque object that the install function can set
 * to serve as a scheduling data structure (e.g. a dequeue, or a
 * priority list).  Only the initializations that cannot be done in
 * parallel should happen at install call. 
 * @return 0 if the scheduler can continue its installation; an error code otherwise
 */
typedef int  (*parsec_sched_base_module_install_fn_t)(parsec_context_t* master);

/**
 * @brief Per-thread Initialization function
 * 
 * @details
 * This call follows the global_init call. It is called once per execution
 * unit, allowing the scheduler to setup any type of local information each stream
 * needs. During this call the scheduler is expected to setup the structures
 * based on the locality information available in the parsec_context_t.
 * eu_context->scheduler_object is a pointer to an opaque structure that
 * the scheduler can define and use to store scheduling information.
 * The barrier provided is global for all execution streams in the 
 * parsec_context_t to which eu_context belongs, and may be used to force
 * synchronizations and setup structures sharing between the different
 * execution streams.
 * @param[inout] eu_context the execution unit that is calling the flow_init
 *               function
 * @param[inout] barrier a barrier common to all execution units in the same
 *               parsec_context_t
 * @return 0 if the scheduler can be used; an error code otherwise
 */
typedef int  (*parsec_sched_base_module_flow_init_fn_t)(parsec_execution_unit_t* eu_context,
                                                       struct parsec_barrier_t* barrier);
/**
 * @brief Scheduling function
 *
 * @details
 * This function, which is called on a given execution stream, is responsible of
 * scheduling the execution of a set of tasks, that are ready to execute.
 *
 * The set of tasks is given through a double linked ring of ready tasks in new_context.
 *
 * The distance argument is a (mandatory) hint, that defines how 'far away' in the
 * list of ready tasks, the new tasks should appear. A higher distance means that
 * tasks should not execute soon, if there are others that are ready, while a smaller
 * distance means that tasks can be considered at any time. A scheduler that does not
 * follow the distance hint may enter a livelock, and try to select tasks that cannot
 * execute. Following the distance hint is critical for fairness of the scheduler, and
 * should not be ignored.
 *
 * A typical scheduler would split the set of tasks in new_context into groups, and
 * store these tasks in the scheduler_object of new_context or other structures
 * accessible from that pointer, for a later selection.
 *
 * @param[inout] eu_context the current execution stream
 * @param[inout] new_context a double-linked ring of ready tasks. Each task has 
 *               a priority field that should be considered as task-specific hints
 *               for the scheduler.
 * @param[in]    distance a (mandatory) hint for the scheduler that enables 
 *               fairness. A higher distance means that the tasks should be selected
 *               late, while a smaller distance means that the tasks can be selected
 *               soon.
 * @return 0 on success; an error code in case of error (which is fatal).
 */
typedef int  (*parsec_sched_base_module_schedule_fn_t)
                 (parsec_execution_unit_t* eu_context,
                  parsec_execution_context_t* new_context,
                  int32_t distance);
/**
 * @brief Selecting Function
 *
 * @details 
 * Select the best candidate to be executed next. This function returns the task to execute,
 * and set the distance where the returned candidate has been found (greater means further away).
 * The distance is more than a hint, if ignored live locks can happen.
 *
 * A typical scheduler would pop the first task from the highest priority list, and provide
 * the priority level of the selected task as distance.
 *
 * @param[inout] eu_context the execution stream that is calling the select function
 * @param[out]   distance the distance from which the task was pulled (ignored if NULL
 *                         is returned
 * @return The selected task, or NULL if none is selectable
 */
typedef parsec_execution_context_t *(*parsec_sched_base_module_select_fn_t)
                 (parsec_execution_unit_t *eu_context,
                  int32_t* distance);

/**
 * @brief Dump runtime statistics.
 *
 * @details
 *   Prints some scheduler-specific runtime statistics on stdout.
 *
 *   This function is called for each execution stream.
 *
 *  @param[in] eu_context the calling execution stream
 */
typedef void (*parsec_sched_base_module_stats_fn_t)(parsec_execution_unit_t* eu_context);

/**
 * @brief Finalization.
 *
 * @details
 * This function is called once for each parsec_context_t, and upon
 * completion the schedulers are supposed to have released all internal
 * resources. Special attention should be taken for destroying the data used by
 * the execution streams as there is no special call for this. However, it should
 * be assumed that when remove is called all existing execution streams have been
 * torn down, and their data can be safely released.
 *
 * @param[inout] master the main parsec_context_t from which to remove the
 *                       scheduler. 
 * @code{c}
 * For each vp in [0; master->nb_vp];
 *     For each eu in [0; master->virtual_process[vp]->nb_vp];
 *         master->virtual_process[vp]->execution_units[eu]->scheduler_object
 * @endcode
 * points to an opaque structure, that is scheduler-specific, and that should
 * be released by this function if the installation / flow_init functions set
 * them.
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

/** @} */

END_C_DECLS

#endif
