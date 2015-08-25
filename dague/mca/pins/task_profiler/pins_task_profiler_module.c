/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <errno.h>
#include <stdio.h>
#include "dague_config.h"
#include "dague/mca/pins/pins.h"
#include "pins_task_profiler.h"
#include "dague/profiling.h"
#include "dague/execution_unit.h"

/* init functions */
static void pins_thread_init_task_profiler(struct dague_execution_unit_s * exec_unit);
static void pins_thread_fini_task_profiler(struct dague_execution_unit_s * exec_unit);

/* PINS callbacks */
static void task_profiler_exec_count_begin(struct dague_execution_unit_s*      exec_unit,
                                           struct dague_execution_context_s*   task,
                                           struct parsec_pins_next_callback_s* cb_data);
static void task_profiler_exec_count_end(struct dague_execution_unit_s*      exec_unit,
                                         struct dague_execution_context_s*   task,
                                         struct parsec_pins_next_callback_s* cb_data);

const dague_pins_module_t dague_pins_task_profiler_module = {
    &dague_pins_task_profiler_component,
    {
        NULL,
        NULL,
        NULL,
        NULL,
        pins_thread_init_task_profiler,
        pins_thread_fini_task_profiler,
    }
};

static void pins_thread_init_task_profiler(struct dague_execution_unit_s * exec_unit)
{
    parsec_pins_next_callback_t* event_cb;
    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PINS_REGISTER(exec_unit, EXEC_BEGIN, task_profiler_exec_count_begin, event_cb);
    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PINS_REGISTER(exec_unit, EXEC_END, task_profiler_exec_count_end, event_cb);
}

static void pins_thread_fini_task_profiler(struct dague_execution_unit_s * exec_unit)
{
    parsec_pins_next_callback_t* event_cb;
    PINS_UNREGISTER(exec_unit, EXEC_BEGIN, task_profiler_exec_count_begin, &event_cb);
    free(event_cb);
    PINS_UNREGISTER(exec_unit, EXEC_END, task_profiler_exec_count_end, &event_cb);
    free(event_cb);
}

/*
 PINS CALLBACKS
 */

static void task_profiler_exec_count_begin(struct dague_execution_unit_s*      exec_unit,
                                           struct dague_execution_context_s*   task,
                                           struct parsec_pins_next_callback_s* cb_data)
{
    if (NULL != task->dague_handle->profiling_array)
        dague_profiling_trace(exec_unit->eu_profile,
                              task->dague_handle->profiling_array[2 * task->function->function_id],
                              (*task->function->key)(task->dague_handle, task->locals),
                              task->dague_handle->handle_id,
                              (void *)NULL);
    (void)cb_data;
}

static void task_profiler_exec_count_end(struct dague_execution_unit_s*      exec_unit,
                                         struct dague_execution_context_s*   task,
                                         struct parsec_pins_next_callback_s* cb_data)
{
    if (NULL != task->dague_handle->profiling_array)
        dague_profiling_trace(exec_unit->eu_profile,
                              task->dague_handle->profiling_array[1 + 2 * task->function->function_id],
                              (*task->function->key)(task->dague_handle, task->locals),
                              task->dague_handle->handle_id,
                              (void *)NULL);
    (void)cb_data;
}
