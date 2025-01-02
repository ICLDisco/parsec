/**
 * Copyright (c) 2016-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/interfaces/interface.h"
#include "parsec/mca/device/device.h"
#include "parsec/utils/debug.h"
#include "parsec/scheduling.h"
#if defined(PARSEC_HAVE_LIMITS_H)
#include <limits.h>
#endif  /* defined(HAVE_LIMITS_H) */

/* Functions to return tasks to their mempool once their execution is
 * completed. The fist one should be used when counting the tasks is
 * not necessary, while the second one contains the task counting.
 */
parsec_hook_return_t
parsec_release_task_to_mempool(parsec_execution_stream_t *es,
                              parsec_task_t *this_task)
{
    (void)es;
    parsec_thread_mempool_free( this_task->mempool_owner, this_task );
    return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
parsec_release_task_to_mempool_update_nbtasks(parsec_execution_stream_t *es,
                                             parsec_task_t *this_task)
{
    parsec_taskpool_t *tp = this_task->taskpool;
    (void)es;
    parsec_thread_mempool_free( this_task->mempool_owner, this_task );
    tp->tdm.module->taskpool_addto_nb_tasks(tp, -1);
    return PARSEC_HOOK_RETURN_DONE;
}

parsec_hook_return_t
parsec_release_task_to_mempool_and_count_as_runtime_tasks(parsec_execution_stream_t *es,
                                                         parsec_task_t *this_task)
{
    parsec_taskpool_t *tp = this_task->taskpool;
    (void)es;
    parsec_thread_mempool_free( this_task->mempool_owner, this_task );
    parsec_taskpool_update_runtime_nbtask(tp, -1);
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
priority_of_generic_startup_as_expr_fct(const parsec_taskpool_t * __tp,
                                        const parsec_assignment_t * locals)
{
    (void)__tp;
    (void)locals;
    return INT_MIN;
}

static const parsec_expr_t priority_of_generic_startup_as_expr = {
    .op = PARSEC_EXPR_OP_INLINE,
    .u_expr.v_func = {.type=PARSEC_RETURN_TYPE_INT32,
                      .func = { .inline_func_int32 = (parsec_expr_op_int32_inline_func_t)priority_of_generic_startup_as_expr_fct}}
};

static inline parsec_key_t
__parsec_generic_startup_make_key(const parsec_taskpool_t *tp, const parsec_assignment_t *as)
{
    (void)as;
    (void)tp;
    return 0;
}

/**
 * This function is a stub that we attach at all the critical locations
 * to make sure the user of these objects are setting them up correcty.
 * The default action here is to assert.
 */
static int
parsec_empty_function_without_arguments(parsec_execution_stream_t *es,
                                        parsec_task_t *this_task)
{
    char tmp[128];
    parsec_fatal("Task %s is incorrectly initialized\n",
                parsec_task_snprintf(tmp, 128, this_task));
    (void)es;
    return PARSEC_HOOK_RETURN_DONE;
}

static const __parsec_chore_t __parsec_generic_startup_chores[] = {
    {.type = PARSEC_DEV_CPU,
     .evaluate = NULL,
     .hook = (parsec_hook_t *) parsec_empty_function_without_arguments},  /* To be replaced at runtime with the correct point to the startup tasks */
    {.type = PARSEC_DEV_NONE,
     .evaluate = NULL,
     .hook = (parsec_hook_t *) NULL},                                     /* End marker */
};

static parsec_key_fn_t __parsec_generic_key_functions = {
    .key_equal = parsec_hash_table_generic_64bits_key_equal,
    .key_print = parsec_hash_table_generic_64bits_key_print,
    .key_hash  = parsec_hash_table_generic_64bits_key_hash
};

const parsec_task_class_t __parsec_generic_startup = {
    .name = "Generic Startup",
    .task_class_id = PARSEC_LOCAL_DATA_TASK_CLASS_ID,  /* To be replaced in all copies */
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
    .make_key = __parsec_generic_startup_make_key,
    .key_functions = &__parsec_generic_key_functions,
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

#if defined(PARSEC_PROF_TRACE)
void *parsec_task_profile_info(void *dst, const void *task_, size_t size)
{
    parsec_task_t *task = (parsec_task_t *)task_;
    return memcpy(dst, &task->prof_info, size);
}
#endif
