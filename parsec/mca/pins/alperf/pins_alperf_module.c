/*
 * Copyright (c) 2018-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/mca/pins/pins.h"
#include "pins_alperf.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/utils/mca_param.h"
#include "parsec/dictionary.h"


/* context level task_classs */
static void pins_init_alperf(parsec_context_t * master_context);
static void pins_fini_alperf(parsec_context_t * master_context);

/* taskpool level functions */
static void pins_taskpool_init_alperf(parsec_taskpool_t * taskpool);
static void pins_taskpool_fini_alperf(parsec_taskpool_t * taskpool);

/* thread level functions */
static void pins_thread_init_alperf(parsec_execution_stream_t * es);
static void pins_thread_fini_alperf(parsec_execution_stream_t * es);

/* PINS callbacks */
static void alperf_exec_count_end(struct parsec_execution_stream_s *es,
                                  struct parsec_task_s *task,
                                  struct parsec_pins_next_callback_s * cb_taskpool);

const parsec_pins_module_t parsec_pins_alperf_module = {
    &parsec_pins_alperf_component,
    {
        pins_init_alperf,
        pins_fini_alperf,
        pins_taskpool_init_alperf,
        pins_taskpool_fini_alperf,
        pins_thread_init_alperf,
        pins_thread_fini_alperf
    },
    { NULL }
};

static void pins_thread_init_alperf(parsec_execution_stream_t *es) {
    parsec_pins_next_callback_t *cb_taskpool = (parsec_pins_next_callback_t*)calloc(1, sizeof(parsec_pins_next_callback_t));
    if (0 > PARSEC_PINS_REGISTER(es, EXEC_END, alperf_exec_count_end, cb_taskpool))
        parsec_warning("alperf PINS module failed registering.");
}

static void pins_thread_fini_alperf(parsec_execution_stream_t *es) {
    parsec_pins_next_callback_t *cb_taskpool;
    PARSEC_PINS_UNREGISTER(es, EXEC_END, alperf_exec_count_end, &cb_taskpool);
    free(cb_taskpool);
}

static void pins_init_alperf(parsec_context_t *master_context) {
    (void)master_context;
    /* Register properties that PaRSEC will expose, be creative */

}

static void pins_fini_alperf(parsec_context_t *master_context) {
    (void)master_context;

}

static void pins_taskpool_init_alperf(parsec_taskpool_t *taskpool) {
    (void)taskpool;

}

static void pins_taskpool_fini_alperf(parsec_taskpool_t *taskpool) {
    (void)taskpool;

}


extern parsec_profiling_dictionary_t *parsec_profiling_dictionary;
/*
 PINS CALLBACKS
 */
static void alperf_exec_count_end(struct parsec_execution_stream_s *es,
                                  struct parsec_task_s *task,
                                  struct parsec_pins_next_callback_s * cb_taskpool) {
    (void)cb_taskpool;
    (void)task;
    (void)es;

    if (parsec_profiling_dictionary->shmem) {
        /* Identify the calling task_class */
        const parsec_profiling_task_class_t *fc
            = parsec_profiling_find_profiling_task_class(task->taskpool->taskpool_name,
                                                         task->task_class->name);
        if (!fc) return; /* Unknown task_class for the profiling subsystem. This should not happen,
                          * however it is safe to just return.
                          */

        /* Let's explore all the properties and evaluate them */
        void *tmp[2] = { (void*)es, (void*)task };
        parsec_hash_table_for_all((parsec_hash_table_t*)&fc->properties, parsec_profiling_evaluate_property, tmp);
    }
}
