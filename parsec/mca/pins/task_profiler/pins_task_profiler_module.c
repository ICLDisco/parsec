/*
 * Copyright (c) 2012-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <errno.h>
#include <stdio.h>
#include "parsec/parsec_config.h"
#include "parsec/mca/pins/pins.h"
#include "pins_task_profiler.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/interfaces/superscalar/insert_function_internal.h"

int release_deps_trace_keyin;
int release_deps_trace_keyout;
int activate_cb_trace_keyin;
int activate_cb_trace_keyout;
int data_flush_trace_keyin;
int data_flush_trace_keyout;

/* init functions */
static void pins_init_task_profiler(parsec_context_t *master_context);
static void pins_fini_task_profiler(parsec_context_t *master_context);
static void pins_thread_init_task_profiler(struct parsec_execution_stream_s * es);
static void pins_thread_fini_task_profiler(struct parsec_execution_stream_s * es);

/* PINS callbacks */
static void task_profiler_release_deps_begin(struct parsec_execution_stream_s*   es,
                                             struct parsec_task_s*               task,
                                             struct parsec_pins_next_callback_s* cb_data);
static void task_profiler_release_deps_end(struct parsec_execution_stream_s*     es,
                                           struct parsec_task_s*                 task,
                                           struct parsec_pins_next_callback_s*   cb_data);
static void task_profiler_activate_cb_begin(struct parsec_execution_stream_s*    es,
                                            struct parsec_task_s*               task,
                                            struct parsec_pins_next_callback_s* cb_data);
static void task_profiler_activate_cb_end(struct parsec_execution_stream_s*     es,
                                          struct parsec_task_s*                 task,
                                          struct parsec_pins_next_callback_s*   cb_data);
static void task_profiler_data_flush_begin(struct parsec_execution_stream_s*    es,
                                           struct parsec_task_s*                task,
                                           struct parsec_pins_next_callback_s*  cb_data);
static void task_profiler_data_flush_end(struct parsec_execution_stream_s*     es,
                                         struct parsec_task_s*                 task,
                                         struct parsec_pins_next_callback_s*   cb_data);

static void task_profiler_exec_count_begin(struct parsec_execution_stream_s*   es,
                                           struct parsec_task_s*               task,
                                           struct parsec_pins_next_callback_s* cb_data);
static void task_profiler_exec_count_end(struct parsec_execution_stream_s*     es,
                                         struct parsec_task_s*                 task,
                                         struct parsec_pins_next_callback_s*   cb_data);

const parsec_pins_module_t parsec_pins_task_profiler_module = {
    &parsec_pins_task_profiler_component,
    {
        pins_init_task_profiler,
        pins_fini_task_profiler,
        NULL,
        NULL,
        pins_thread_init_task_profiler,
        pins_thread_fini_task_profiler
    },
    { NULL }
};

static void pins_init_task_profiler(parsec_context_t *master_context)
{
    (void)master_context;
    parsec_profiling_add_dictionary_keyword("RELEASE_DEPS", "fill:#FF0000",
                                           sizeof(int32_t),
                                           "rd_fid{int32_t}",
                                           &release_deps_trace_keyin,
                                           &release_deps_trace_keyout);

    parsec_profiling_add_dictionary_keyword("ACTIVATE_CB", "fill:#FFF000",
                                           0,
                                           "",
                                           &activate_cb_trace_keyin,
                                           &activate_cb_trace_keyout);

    parsec_profiling_add_dictionary_keyword("DATA_FLUSH", "fill:#FFF000",
                                           0,
                                           "",
                                           &data_flush_trace_keyin,
                                           &data_flush_trace_keyout);
}

static void pins_fini_task_profiler(parsec_context_t *master_context)
{
    (void)master_context;
}

static void pins_thread_init_task_profiler(struct parsec_execution_stream_s * es)
{
    parsec_pins_next_callback_t* event_cb;
    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PARSEC_PINS_REGISTER(es, EXEC_BEGIN, task_profiler_exec_count_begin, event_cb);
    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PARSEC_PINS_REGISTER(es, EXEC_END, task_profiler_exec_count_end, event_cb);

    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PARSEC_PINS_REGISTER(es, RELEASE_DEPS_BEGIN, task_profiler_release_deps_begin, event_cb);
    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PARSEC_PINS_REGISTER(es, RELEASE_DEPS_END, task_profiler_release_deps_end, event_cb);

    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PARSEC_PINS_REGISTER(es, ACTIVATE_CB_BEGIN, task_profiler_activate_cb_begin, event_cb);
    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PARSEC_PINS_REGISTER(es, ACTIVATE_CB_END, task_profiler_activate_cb_end, event_cb);

    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PARSEC_PINS_REGISTER(es, DATA_FLUSH_BEGIN, task_profiler_data_flush_begin, event_cb);
    event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
    PARSEC_PINS_REGISTER(es, DATA_FLUSH_END, task_profiler_data_flush_end, event_cb);
}

static void pins_thread_fini_task_profiler(struct parsec_execution_stream_s * es)
{
    parsec_pins_next_callback_t* event_cb;
    PARSEC_PINS_UNREGISTER(es, EXEC_BEGIN, task_profiler_exec_count_begin, &event_cb);
    free(event_cb);
    PARSEC_PINS_UNREGISTER(es, EXEC_END, task_profiler_exec_count_end, &event_cb);
    free(event_cb);

    PARSEC_PINS_UNREGISTER(es, RELEASE_DEPS_BEGIN, task_profiler_release_deps_begin, &event_cb);
    free(event_cb);
    PARSEC_PINS_UNREGISTER(es, RELEASE_DEPS_END, task_profiler_release_deps_end, &event_cb);
    free(event_cb);

    PARSEC_PINS_UNREGISTER(es, ACTIVATE_CB_BEGIN, task_profiler_activate_cb_begin, &event_cb);
    free(event_cb);
    PARSEC_PINS_UNREGISTER(es, ACTIVATE_CB_END, task_profiler_activate_cb_end, &event_cb);
    free(event_cb);

    PARSEC_PINS_UNREGISTER(es, DATA_FLUSH_BEGIN, task_profiler_data_flush_begin, &event_cb);
    free(event_cb);
    PARSEC_PINS_UNREGISTER(es, DATA_FLUSH_END, task_profiler_data_flush_end, &event_cb);
    free(event_cb);
}

/*
 PINS CALLBACKS
 */

static void
task_profiler_release_deps_begin(struct parsec_execution_stream_s*   es,
                                 struct parsec_task_s*               task,
                                 struct parsec_pins_next_callback_s* cb_data)
{
    int32_t rd_fid = task->task_class->task_class_id;

    PARSEC_PROFILING_TRACE(es->es_profile,
                           release_deps_trace_keyin,
                           task->task_class->key_functions->key_hash(task->task_class->make_key(task->taskpool, task->locals), NULL),
                           task->taskpool->taskpool_id,
                           (void *)&rd_fid);

    (void)cb_data;
}

static void
task_profiler_release_deps_end(struct parsec_execution_stream_s*   es,
                               struct parsec_task_s*               task,
                               struct parsec_pins_next_callback_s* cb_data)
{
    int32_t rd_fid = task->task_class->task_class_id;

    PARSEC_PROFILING_TRACE(es->es_profile,
                           release_deps_trace_keyout,
                           task->task_class->key_functions->key_hash(task->task_class->make_key(task->taskpool, task->locals), NULL),
                           task->taskpool->taskpool_id,
                           (void*)&rd_fid);
    (void)cb_data;
}

static void
task_profiler_activate_cb_begin(struct parsec_execution_stream_s*   es,
                                struct parsec_task_s*               task,
                                struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           activate_cb_trace_keyin,
                           0,
                           -1,
                           NULL);

    (void)cb_data;(void)task;
}

static void
task_profiler_activate_cb_end(struct parsec_execution_stream_s*   es,
                              struct parsec_task_s*               task,
                              struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           activate_cb_trace_keyout,
                           0,
                           -1,
                           NULL);
    (void)cb_data;(void)task;
}

static void
task_profiler_data_flush_begin(struct parsec_execution_stream_s*   es,
                               struct parsec_task_s*               task,
                               struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           data_flush_trace_keyin,
                           0,
                           -1,
                           NULL);

    (void)cb_data;(void)task;
}

static void
task_profiler_data_flush_end(struct parsec_execution_stream_s*   es,
                             struct parsec_task_s*               task,
                             struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           data_flush_trace_keyout,
                           0,
                           -1,
                           NULL);
    (void)cb_data;(void)task;
}

static void
task_profiler_exec_count_begin(struct parsec_execution_stream_s*   es,
                               struct parsec_task_s*               task,
                               struct parsec_pins_next_callback_s* cb_data)
{
    if (NULL != task->taskpool->profiling_array)
        PARSEC_PROFILING_TRACE(es->es_profile,
                               task->taskpool->profiling_array[2 * task->task_class->task_class_id],
                               task->task_class->key_functions->key_hash(task->task_class->make_key(task->taskpool, task->locals), NULL),
                               task->taskpool->taskpool_id,
                               (void *)NULL);
    (void)cb_data;
}

static void
task_profiler_exec_count_end(struct parsec_execution_stream_s*   es,
                             struct parsec_task_s*               task,
                             struct parsec_pins_next_callback_s* cb_data)
{
    if (NULL != task->taskpool->profiling_array)
        PARSEC_PROFILING_TRACE(es->es_profile,
                               task->taskpool->profiling_array[1 + 2 * task->task_class->task_class_id],
                               task->task_class->key_functions->key_hash(task->task_class->make_key(task->taskpool, task->locals), NULL),
                               task->taskpool->taskpool_id,
                               (void *)NULL);
    (void)cb_data;
}

