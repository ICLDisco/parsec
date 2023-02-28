/*
 * Copyright (c) 2012-2023 The University of Tennessee and The University
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
#include "parsec/interfaces/dtd/insert_function_internal.h"
#include "parsec/parsec_binary_profile.h"
#include "parsec/utils/argv.h"
#include "parsec/utils/mca_param.h"

static int trace_keys[PARSEC_PINS_FLAG_COUNT] = {0};

static char *mca_param_string;

/* init functions */
static void pins_init_task_profiler(parsec_context_t *master_context);
static void pins_fini_task_profiler(parsec_context_t *master_context);
static void pins_thread_init_task_profiler(struct parsec_execution_stream_s * es);
static void pins_thread_fini_task_profiler(struct parsec_execution_stream_s * es);

/* PINS callbacks */
static void task_profiler_prepare_input_begin(struct parsec_execution_stream_s*   es,
                                              struct parsec_task_s*               task,
                                              struct parsec_pins_next_callback_s* cb_data);
static void task_profiler_prepare_input_end(struct parsec_execution_stream_s*     es,
                                            struct parsec_task_s*                 task,
                                            struct parsec_pins_next_callback_s*   cb_data);
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
static void task_profiler_select_begin(struct parsec_execution_stream_s*    es,
                                       struct parsec_task_s*                task,
                                       struct parsec_pins_next_callback_s*  cb_data);
static void task_profiler_select_end(struct parsec_execution_stream_s*     es,
                                     struct parsec_task_s*                 task,
                                     struct parsec_pins_next_callback_s*   cb_data);
static void task_profiler_complete_exec_begin(struct parsec_execution_stream_s*    es,
                                              struct parsec_task_s*                task,
                                              struct parsec_pins_next_callback_s*  cb_data);
static void task_profiler_complete_exec_end(struct parsec_execution_stream_s*     es,
                                            struct parsec_task_s*                 task,
                                            struct parsec_pins_next_callback_s*   cb_data);
static void task_profiler_schedule_begin(struct parsec_execution_stream_s*    es,
                                         struct parsec_task_s*                task,
                                         struct parsec_pins_next_callback_s*  cb_data);
static void task_profiler_schedule_end(struct parsec_execution_stream_s*     es,
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

    parsec_mca_param_reg_string_name("pins", "task_profiler_event",
                                     "Comma-separated list of task profiler events to be gathered.\n",
                                     false, false,
                                     parsec_pins_enable_default_names, &mca_param_string);
    char **events = parsec_argv_split(mca_param_string, ',');
    int i = 0;
    while (events[i] != NULL) {
        char *event = events[i];
        PARSEC_PINS_FLAG flag = parsec_pins_name_to_begin_flag(event);
        if (flag < PARSEC_PINS_FLAG_COUNT) {
            parsec_pins_enable_mask |= flag;
        }
        free(event);
        ++i;
    }
    free(events);

    if (PARSEC_PINS_FLAG_ENABLED(PREPARE_INPUT_BEGIN)) {
        parsec_profiling_add_dictionary_keyword("PARSEC RUNTIME::PREPARE_INPUT", "fill:#FF0000",
                                                0,
                                                "",
                                                &trace_keys[PREPARE_INPUT_BEGIN],
                                                &trace_keys[PREPARE_INPUT_END]);
    }

    if (PARSEC_PINS_FLAG_ENABLED(RELEASE_DEPS_BEGIN)) {
        parsec_profiling_add_dictionary_keyword("PARSEC RUNTIME::RELEASE_DEPS", "fill:#FF0000",
                                                sizeof(int32_t),
                                                "tcid{uint32_t}",
                                                &trace_keys[RELEASE_DEPS_BEGIN],
                                                &trace_keys[RELEASE_DEPS_END]);
    }

    if (PARSEC_PINS_FLAG_ENABLED(ACTIVATE_CB_BEGIN)) {
        parsec_profiling_add_dictionary_keyword("PARSEC RUNTIME::ACTIVATE_CB", "fill:#FFF000",
                                                0,
                                                "",
                                                &trace_keys[ACTIVATE_CB_BEGIN],
                                                &trace_keys[ACTIVATE_CB_END]);
    }

    if (PARSEC_PINS_FLAG_ENABLED(DATA_FLUSH_BEGIN)) {
        parsec_profiling_add_dictionary_keyword("PARSEC RUNTIME::DATA_FLUSH", "fill:#FFF000",
                                                0,
                                                "",
                                                &trace_keys[DATA_FLUSH_BEGIN],
                                                &trace_keys[DATA_FLUSH_END]);
    }

    if (PARSEC_PINS_FLAG_ENABLED(SELECT_BEGIN)) {
        parsec_profiling_add_dictionary_keyword("PARSEC RUNTIME::SELECT_TASK", "fill:#EFEFEF",
                                                0,
                                                "",
                                                &trace_keys[SELECT_BEGIN],
                                                &trace_keys[SELECT_END]);
    }

    if (PARSEC_PINS_FLAG_ENABLED(SCHEDULE_BEGIN)) {
        parsec_profiling_add_dictionary_keyword("PARSEC RUNTIME::SCHEDULE_TASKS", "fill:#EFEFEF",
                                                0,
                                                "",
                                                &trace_keys[SCHEDULE_BEGIN],
                                                &trace_keys[SCHEDULE_END]);
    }

    if (PARSEC_PINS_FLAG_ENABLED(COMPLETE_EXEC_BEGIN)) {
        parsec_profiling_add_dictionary_keyword("PARSEC RUNTIME::COMPLETE_EXEC", "fill:#EFEFEF",
                                                0,
                                                "",
                                                &trace_keys[COMPLETE_EXEC_BEGIN],
                                                &trace_keys[COMPLETE_EXEC_END]);
    }
}

static void pins_fini_task_profiler(parsec_context_t *master_context)
{
    (void)master_context;
}

static void pins_thread_init_task_profiler(struct parsec_execution_stream_s * es)
{
    parsec_pins_next_callback_t* event_cb;

    if (PARSEC_PINS_FLAG_ENABLED(PREPARE_INPUT_BEGIN)) {
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, PREPARE_INPUT_BEGIN, task_profiler_prepare_input_begin, event_cb);
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, PREPARE_INPUT_END, task_profiler_prepare_input_end, event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(EXEC_BEGIN)) {
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, EXEC_BEGIN, task_profiler_exec_count_begin, event_cb);
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, EXEC_END, task_profiler_exec_count_end, event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(RELEASE_DEPS_BEGIN)) {
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, RELEASE_DEPS_BEGIN, task_profiler_release_deps_begin, event_cb);
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, RELEASE_DEPS_END, task_profiler_release_deps_end, event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(ACTIVATE_CB_BEGIN)) {
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, ACTIVATE_CB_BEGIN, task_profiler_activate_cb_begin, event_cb);
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, ACTIVATE_CB_END, task_profiler_activate_cb_end, event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(DATA_FLUSH_BEGIN)) {
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, DATA_FLUSH_BEGIN, task_profiler_data_flush_begin, event_cb);
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, DATA_FLUSH_END, task_profiler_data_flush_end, event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(SELECT_BEGIN)) {
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, SELECT_BEGIN, task_profiler_select_begin, event_cb);
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, SELECT_END, task_profiler_select_end, event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(SCHEDULE_BEGIN)) {
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, SCHEDULE_BEGIN, task_profiler_schedule_begin, event_cb);
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, SCHEDULE_END, task_profiler_schedule_end, event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(COMPLETE_EXEC_BEGIN)) {
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, COMPLETE_EXEC_BEGIN, task_profiler_complete_exec_begin, event_cb);
        event_cb = (parsec_pins_next_callback_t*)malloc(sizeof(parsec_pins_next_callback_t));
        PARSEC_PINS_REGISTER(es, COMPLETE_EXEC_END, task_profiler_complete_exec_end, event_cb);
    }
}

static void pins_thread_fini_task_profiler(struct parsec_execution_stream_s * es)
{
    parsec_pins_next_callback_t* event_cb;

    if (PARSEC_PINS_FLAG_ENABLED(PREPARE_INPUT_BEGIN)) {
        PARSEC_PINS_UNREGISTER(es, PREPARE_INPUT_BEGIN, task_profiler_prepare_input_begin, &event_cb);
        free(event_cb);
        PARSEC_PINS_UNREGISTER(es, PREPARE_INPUT_END, task_profiler_prepare_input_end, &event_cb);
        free(event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(EXEC_BEGIN)) {
        PARSEC_PINS_UNREGISTER(es, EXEC_BEGIN, task_profiler_exec_count_begin, &event_cb);
        free(event_cb);
        PARSEC_PINS_UNREGISTER(es, EXEC_END, task_profiler_exec_count_end, &event_cb);
        free(event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(RELEASE_DEPS_BEGIN)) {
        PARSEC_PINS_UNREGISTER(es, RELEASE_DEPS_BEGIN, task_profiler_release_deps_begin, &event_cb);
        free(event_cb);
        PARSEC_PINS_UNREGISTER(es, RELEASE_DEPS_END, task_profiler_release_deps_end, &event_cb);
        free(event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(ACTIVATE_CB_BEGIN)) {
        PARSEC_PINS_UNREGISTER(es, ACTIVATE_CB_BEGIN, task_profiler_activate_cb_begin, &event_cb);
        free(event_cb);
        PARSEC_PINS_UNREGISTER(es, ACTIVATE_CB_END, task_profiler_activate_cb_end, &event_cb);
        free(event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(DATA_FLUSH_BEGIN)) {
        PARSEC_PINS_UNREGISTER(es, DATA_FLUSH_BEGIN, task_profiler_data_flush_begin, &event_cb);
        free(event_cb);
        PARSEC_PINS_UNREGISTER(es, DATA_FLUSH_END, task_profiler_data_flush_end, &event_cb);
        free(event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(SELECT_BEGIN)) {
        PARSEC_PINS_UNREGISTER(es, SELECT_BEGIN, task_profiler_select_begin, &event_cb);
        free(event_cb);
        PARSEC_PINS_UNREGISTER(es, SELECT_END, task_profiler_select_end, &event_cb);
        free(event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(SCHEDULE_BEGIN)) {
        PARSEC_PINS_UNREGISTER(es, SCHEDULE_BEGIN, task_profiler_schedule_begin, &event_cb);
        free(event_cb);
        PARSEC_PINS_UNREGISTER(es, SCHEDULE_END, task_profiler_schedule_end, &event_cb);
        free(event_cb);
    }

    if (PARSEC_PINS_FLAG_ENABLED(COMPLETE_EXEC_BEGIN)) {
        PARSEC_PINS_UNREGISTER(es, COMPLETE_EXEC_BEGIN, task_profiler_complete_exec_begin, &event_cb);
        free(event_cb);
        PARSEC_PINS_UNREGISTER(es, COMPLETE_EXEC_END, task_profiler_complete_exec_end, &event_cb);
        free(event_cb);
    }
}

/*
 PINS CALLBACKS
 */
static void
task_profiler_prepare_input_begin(struct parsec_execution_stream_s*   es,
                                 struct parsec_task_s*               task,
                                 struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           trace_keys[PREPARE_INPUT_BEGIN],
                           0,
                           -1,
                           NULL);

    (void)cb_data;(void)task;
}

static void
task_profiler_prepare_input_end(struct parsec_execution_stream_s*   es,
                               struct parsec_task_s*               task,
                               struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           trace_keys[PREPARE_INPUT_END],
                           0,
                           -1,
                           NULL);
    (void)cb_data;(void)task;
}


static void
task_profiler_release_deps_begin(struct parsec_execution_stream_s*   es,
                                 struct parsec_task_s*               task,
                                 struct parsec_pins_next_callback_s* cb_data)
{
    int32_t tcid = task->task_class->task_class_id;

    PARSEC_PROFILING_TRACE(es->es_profile,
                           trace_keys[RELEASE_DEPS_BEGIN],
                           task->task_class->key_functions->key_hash(task->task_class->make_key(task->taskpool, task->locals), NULL),
                           task->taskpool->taskpool_id,
                           (void *)&tcid);

    (void)cb_data;
}

static void
task_profiler_release_deps_end(struct parsec_execution_stream_s*   es,
                               struct parsec_task_s*               task,
                               struct parsec_pins_next_callback_s* cb_data)
{
    int32_t tcid = task->task_class->task_class_id;

    PARSEC_PROFILING_TRACE(es->es_profile,
                           trace_keys[RELEASE_DEPS_END],
                           task->task_class->key_functions->key_hash(task->task_class->make_key(task->taskpool, task->locals), NULL),
                           task->taskpool->taskpool_id,
                           (void*)&tcid);
    (void)cb_data;
}

static void
task_profiler_activate_cb_begin(struct parsec_execution_stream_s*   es,
                                struct parsec_task_s*               task,
                                struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           trace_keys[ACTIVATE_CB_BEGIN],
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
                           trace_keys[ACTIVATE_CB_END],
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
                           trace_keys[DATA_FLUSH_BEGIN],
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
                           trace_keys[DATA_FLUSH_END],
                           0,
                           -1,
                           NULL);
    (void)cb_data;(void)task;
}

static void
task_profiler_select_begin(struct parsec_execution_stream_s*   es,
                           struct parsec_task_s*               task,
                           struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           trace_keys[SELECT_BEGIN],
                           0,
                           -1,
                           NULL);

    (void)cb_data;(void)task;
}

static void
task_profiler_select_end(struct parsec_execution_stream_s*   es,
                         struct parsec_task_s*               task,
                         struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           trace_keys[SELECT_END],
                           0,
                           -1,
                           NULL);
    (void)cb_data;(void)task;
}

static void
task_profiler_schedule_begin(struct parsec_execution_stream_s*   es,
                            struct parsec_task_s*               task,
                            struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           trace_keys[SCHEDULE_BEGIN],
                           0,
                           -1,
                           NULL);

    (void)cb_data;(void)task;
}

static void
task_profiler_schedule_end(struct parsec_execution_stream_s*   es,
                           struct parsec_task_s*               task,
                           struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           trace_keys[SCHEDULE_END],
                           0,
                           -1,
                           NULL);
    (void)cb_data;(void)task;
}

static void
task_profiler_complete_exec_begin(struct parsec_execution_stream_s*   es,
                                  struct parsec_task_s*               task,
                                  struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           trace_keys[COMPLETE_EXEC_BEGIN],
                           0,
                           -1,
                           NULL);

    (void)cb_data;(void)task;
}

static void
task_profiler_complete_exec_end(struct parsec_execution_stream_s*   es,
                                struct parsec_task_s*               task,
                                struct parsec_pins_next_callback_s* cb_data)
{
    PARSEC_PROFILING_TRACE(es->es_profile,
                           trace_keys[COMPLETE_EXEC_END],
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
    if (NULL != task->taskpool->profiling_array &&
        task->task_class->task_class_id < task->taskpool->nb_task_classes)
        PARSEC_TASK_PROF_TRACE_FLAGS(es->es_profile,
                               task->taskpool->profiling_array[START_KEY(task->task_class->task_class_id)],
                               task,
                               PARSEC_PROFILING_EVENT_TIME_AT_END, 0);
    (void)cb_data;
}

static void
task_profiler_exec_count_end(struct parsec_execution_stream_s*   es,
                             struct parsec_task_s*               task,
                             struct parsec_pins_next_callback_s* cb_data)
{
    if (NULL != task->taskpool->profiling_array &&
        task->task_class->task_class_id < task->taskpool->nb_task_classes)
        PARSEC_TASK_PROF_TRACE_FLAGS(es->es_profile,
                               task->taskpool->profiling_array[END_KEY(task->task_class->task_class_id)],
                               task,
                               PARSEC_PROFILING_EVENT_TIME_AT_START, 1);
    (void)cb_data;
}

