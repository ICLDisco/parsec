/**
 * Copyright (c) 2016      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef DAGUE_INTERFACE_H_HAS_BEEN_INCLUDED
#define DAGUE_INTERFACE_H_HAS_BEEN_INCLUDED

#include "dague_config.h"
#include "dague/dague_internal.h"

/**
 * Generic startup function for DSLs. For more info read comment in
 * dague/interface/interface.c
 */
DAGUE_DECLSPEC extern const dague_function_t __dague_generic_startup;

/* Functions to return tasks to their mempool once their execution is
 * completed. The fist one should be used when counting the tasks is
 * not necessary, while the second one contains the task counting.
 */
dague_hook_return_t
dague_release_task_to_mempool(dague_execution_unit_t *eu,
                              dague_execution_context_t *this_task);
dague_hook_return_t
dague_release_task_to_mempool_update_nbtasks(dague_execution_unit_t *eu,
                                             dague_execution_context_t *this_task);

dague_hook_return_t
dague_release_task_to_mempool_and_count_as_runtime_tasks(dague_execution_unit_t *eu,
                                                         dague_execution_context_t *this_task);
#endif  /* DAGUE_INTERFACE_H_HAS_BEEN_INCLUDED */
