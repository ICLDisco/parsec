/**
 * Copyright (c) 2016-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec_config.h"
#include "parsec/interfaces/interface.h"
#include "parsec/devices/device.h"
#include "parsec/debug.h"
#include "parsec/scheduling.h"
#if defined(PARSEC_HAVE_LIMITS_H)
#include <limits.h>
#endif  /* defined(HAVE_LIMITS_H) */

/* Functions to return tasks to their mempool once their execution is
 * completed. The fist one should be used when counting the tasks is
 * not necessary, while the second one contains the task counting.
 */
parsec_hook_return_t
parsec_release_task_to_mempool(parsec_execution_unit_t *eu,
                              parsec_execution_context_t *this_task)
{
    (void)eu;
    parsec_thread_mempool_free( this_task->mempool_owner, this_task );
    return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
parsec_release_task_to_mempool_update_nbtasks(parsec_execution_unit_t *eu,
                                             parsec_execution_context_t *this_task)
{
    parsec_handle_t *handle;
    (void)eu;
    handle = this_task->parsec_handle;
    parsec_thread_mempool_free( this_task->mempool_owner, this_task );
    (void)parsec_atomic_dec_32b( (uint32_t*)&handle->nb_tasks );
    return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
parsec_release_task_to_mempool_and_count_as_runtime_tasks(parsec_execution_unit_t *eu,
                                                         parsec_execution_context_t *this_task)
{
    parsec_handle_t *handle;
    (void)eu;
    handle = this_task->parsec_handle;
    parsec_thread_mempool_free( this_task->mempool_owner, this_task );
    parsec_handle_update_runtime_nbtask(handle, -1);
    return PARSEC_HOOK_RETURN_DONE;
}

/**
 * Special function for generating the dynamic startup functions. The startup
 * code will be of the type of this function, allowing the PaRSEC infrastructure
 * to work. Instead of generating a list of ready tasks during the enqueue and
 * then struggle to spread these tasks over the available resources, we can
 * simply generate a single task (of this type), and allow the runtime to
 * schedule it at its convenience. During the execution of this special task we
 * will generate all the startup tasks, in a manner similar to what we did
 * before. However, we now have the opportunity to build a stateful reentrant
 * function, that can generate only a certain number of tasks and then
 * re-enqueue itself for later execution.
 */

static inline int
priority_of_generic_startup_as_expr_fct(const parsec_handle_t * __parsec_handle,
                                        const assignment_t * locals)
{
    (void)__parsec_handle;
    (void)locals;
    return INT_MIN;
}

static const expr_t priority_of_generic_startup_as_expr = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = (expr_op_int32_inline_func_t)priority_of_generic_startup_as_expr_fct}
};

static inline uint64_t
__parsec_generic_startup_hash(const parsec_handle_t * __parsec_handle,
                             const assignment_t * assignments)
{
    (void)__parsec_handle;
    (void)assignments;
    return 0ULL;
}

/**
 * This function is a stub that we attach at all the critical locations
 * to make sure the user of these objects are setting them up correcty.
 * The default action here is to assert.
 */
static int parsec_empty_function_without_arguments(parsec_execution_unit_t *eu,
                                                  parsec_execution_context_t *this_task)
{
    char tmp[128];
    parsec_fatal("Task %s is incorrectly initialized\n",
                parsec_snprintf_execution_context(tmp, 128, this_task));
    (void)eu;
    return PARSEC_HOOK_RETURN_DONE;
}

static const __parsec_chore_t __parsec_generic_startup_chores[] = {
    {.type = PARSEC_DEV_CPU,
     .evaluate = NULL,
     .hook = (parsec_hook_t *) parsec_empty_function_without_arguments},  /* To be replaced at runtime with the correct point to the startup tasks */
    {.type = PARSEC_DEV_NONE,
     .evaluate = NULL,
     .hook = (parsec_hook_t *) NULL},	/* End marker */
};

const parsec_function_t __parsec_generic_startup = {
    .name = "Generic Startup",
    .function_id = -1,  /* To be replaced in all copies */
    .nb_flows = 0,
    .nb_parameters = 0,
    .nb_locals = 0,
    .params = {NULL},
    .locals = {NULL},
    .data_affinity = (parsec_data_ref_fn_t *) NULL,
    .initial_data = (parsec_data_ref_fn_t *) NULL,
    .final_data = (parsec_data_ref_fn_t *) NULL,
    .priority = &priority_of_generic_startup_as_expr,
    .in = {NULL},
    .out = {NULL},
    .flags = PARSEC_USE_DEPS_MASK,
    .dependencies_goal = 0x0,
    .key = (parsec_functionkey_fn_t *) __parsec_generic_startup_hash,
    .fini = (parsec_hook_t *) NULL,
    .incarnations = __parsec_generic_startup_chores,
    .iterate_successors = (parsec_traverse_function_t *) NULL,
    .iterate_predecessors = (parsec_traverse_function_t *) NULL,
    .release_deps = (parsec_release_deps_t *) NULL,
    .prepare_input = (parsec_hook_t *) parsec_empty_function_without_arguments,
    .prepare_output = (parsec_hook_t *) NULL,
    .complete_execution = (parsec_hook_t *)NULL,
    .release_task = (parsec_hook_t *) parsec_release_task_to_mempool_update_nbtasks,
#if defined(PARSEC_SIM)
    .sim_cost_fct = (parsec_sim_cost_fct_t *) NULL,
#endif
};
