/**
 * Copyright (c) 2016-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PARSEC_INTERFACE_H_HAS_BEEN_INCLUDED
#define PARSEC_INTERFACE_H_HAS_BEEN_INCLUDED

#include "parsec/parsec_internal.h"

BEGIN_C_DECLS

/**
 * @addtogroup parsec_internal_runtime
 *  @{
 */

/**
 * Task class ID for dependencies targeting local data (used as input).
 * Use this as the task_class_id of the dep_t to pinpoint to a local
 * data instead of a task.
 */
#define PARSEC_LOCAL_DATA_TASK_CLASS_ID ((uint8_t)0xFF) /* task_class_id is uint8_t */

/**
 * Generic startup function for DSLs. For more info read comment in
 * parsec/interface/interface.c
 */
PARSEC_DECLSPEC extern const parsec_task_class_t __parsec_generic_startup;

/**
 * Function to return tasks to their mempool once their execution is
 * completed. This function should be used when counting the number of
 * tasks active in the taskpool (that hosts these tasks) is not
 * necessary (e.g. when the number of tasks is undetermined, when
 * completion of a taskpool is triggered internally by task execution
 * or when the runtime system automatically detects the termination).
 */
parsec_hook_return_t
parsec_release_task_to_mempool(parsec_execution_stream_t *es,
                               parsec_task_t *this_task);

/**
 * Function to return tasks to their mempool once their execution is
 * completed. This function should be used when counting the number of
 * active tasks in their taskpool is used to detect termination.
 */
parsec_hook_return_t
parsec_release_task_to_mempool_update_nbtasks(parsec_execution_stream_t *es,
                                              parsec_task_t *this_task);

/**
 * Function to return tasks to their mempool once their execution is
 * completed. This function should be used when a task should be counted
 * as an internal runtime task and not as a user task belonging to the taskpool.
 */
parsec_hook_return_t
parsec_release_task_to_mempool_and_count_as_runtime_tasks(parsec_execution_stream_t *es,
                                                          parsec_task_t *this_task);

/**
 * @}
 */

END_C_DECLS

#endif  /* PARSEC_INTERFACE_H_HAS_BEEN_INCLUDED */
