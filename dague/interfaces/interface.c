/**
 * Copyright (c) 2016      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague/interfaces/interface.h"
#include "dague/devices/device.h"
#include "dague/debug.h"
#include "dague/scheduling.h"
#if defined(DAGUE_HAVE_LIMITS_H)
#include <limits.h>
#endif  /* defined(HAVE_LIMITS_H) */

/* Functions to return tasks to their mempool once their execution is
 * completed. The fist one should be used when counting the tasks is
 * not necessary, while the second one contains the task counting.
 */
dague_hook_return_t
dague_release_task_to_mempool(dague_execution_unit_t *eu,
                              dague_execution_context_t *this_task)
{
    (void)eu;
    dague_thread_mempool_free( this_task->super.mempool_owner, this_task );
    return DAGUE_HOOK_RETURN_DONE;
}

dague_hook_return_t
dague_release_task_to_mempool_update_nbtasks(dague_execution_unit_t *eu,
                                             dague_execution_context_t *this_task)
{
    dague_handle_t *handle;
    (void)eu;
    handle = this_task->dague_handle;
    dague_thread_mempool_free( this_task->super.mempool_owner, this_task );
    dague_atomic_dec_32b( (uint32_t*)&handle->nb_tasks );
    return DAGUE_HOOK_RETURN_DONE;
}

dague_hook_return_t
dague_release_task_to_mempool_and_count_as_runtime_tasks(dague_execution_unit_t *eu,
                                                         dague_execution_context_t *this_task)
{
    dague_handle_t *handle;
    (void)eu;
    handle = this_task->dague_handle;
    dague_thread_mempool_free( this_task->super.mempool_owner, this_task );
    dague_handle_update_runtime_nbtask(handle, -1);
    return DAGUE_HOOK_RETURN_DONE;
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
priority_of_generic_startup_as_expr_fct(const dague_handle_t * __dague_handle,
                                        const assignment_t * locals)
{
    (void)__dague_handle;
    (void)locals;
    return INT_MIN;
}

static const expr_t priority_of_generic_startup_as_expr = {
    .op = EXPR_OP_INLINE,
    .u_expr = {.inline_func_int32 = (expr_op_int32_inline_func_t)priority_of_generic_startup_as_expr_fct}
};

static inline uint64_t
__dague_generic_startup_hash(const dague_handle_t * __dague_handle,
                             const assignment_t * assignments)
{
    (void)__dague_handle;
    (void)assignments;
    return 0ULL;
}

/**
 * This function is a stub that we attach at all the critical locations
 * to make sure the user of these objects are setting them up correcty.
 * The default action here is to assert.
 */
static int dague_empty_function_without_arguments(dague_execution_unit_t *eu,
                                                  dague_execution_context_t *this_task)
{
    char tmp[128];
    dague_abort("Task %s is incorrectly initialized\n",
                dague_snprintf_execution_context(tmp, 128, this_task));
    (void)eu;
    return DAGUE_HOOK_RETURN_DONE;
}

static const __dague_chore_t __dague_generic_startup_chores[] = {
    {.type = DAGUE_DEV_CPU,
     .evaluate = NULL,
     .hook = (dague_hook_t *) dague_empty_function_without_arguments},  /* To be replaced at runtime with the correct point to the startup tasks */
    {.type = DAGUE_DEV_NONE,
     .evaluate = NULL,
     .hook = (dague_hook_t *) NULL},	/* End marker */
};

const dague_function_t __dague_generic_startup = {
    .name = "Generic Startup",
    .function_id = -1,  /* To be replaced in all copies */
    .nb_flows = 0,
    .nb_parameters = 0,
    .nb_locals = 0,
    .params = {NULL},
    .locals = {NULL},
    .data_affinity = (dague_data_ref_fn_t *) NULL,
    .initial_data = (dague_data_ref_fn_t *) NULL,
    .final_data = (dague_data_ref_fn_t *) NULL,
    .priority = &priority_of_generic_startup_as_expr,
    .in = {NULL},
    .out = {NULL},
    .flags = DAGUE_USE_DEPS_MASK,
    .dependencies_goal = 0x0,
    .key = (dague_functionkey_fn_t *) __dague_generic_startup_hash,
    .fini = (dague_hook_t *) NULL,
    .incarnations = __dague_generic_startup_chores,
    .iterate_successors = (dague_traverse_function_t *) NULL,
    .iterate_predecessors = (dague_traverse_function_t *) NULL,
    .release_deps = (dague_release_deps_t *) NULL,
    .prepare_input = (dague_hook_t *) dague_empty_function_without_arguments,
    .prepare_output = (dague_hook_t *) NULL,
    .complete_execution = (dague_hook_t *)NULL,
    .release_task = (dague_hook_t *) dague_release_task_to_mempool_update_nbtasks,
#if defined(DAGUE_SIM)
    .sim_cost_fct = (dague_sim_cost_fct_t *) NULL,
#endif
};
