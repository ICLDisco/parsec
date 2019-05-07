/*
 * Copyright (c) 2013-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include "parsec/mca/pins/pins.h"
#include "pins_iterators_checker.h"
#include "parsec/profiling.h"
#include "parsec/execution_stream.h"
#include "parsec/data_internal.h"
#include "parsec/parsec_internal.h"

#include <errno.h>
#include <stdio.h>

/* init functions */
static void pins_thread_init_iterators_checker(struct parsec_execution_stream_s* es);
static void pins_thread_fini_iterators_checker(struct parsec_execution_stream_s* es);

/* PINS callbacks */
static void
iterators_checker_exec_count_begin(parsec_execution_stream_t* es,
                                   parsec_task_t* task,
                                   struct parsec_pins_next_callback_s* data);

const parsec_pins_module_t parsec_pins_iterators_checker_module = {
    &parsec_pins_iterators_checker_component,
    {
        NULL,
        NULL,
        NULL,
        NULL,
        pins_thread_init_iterators_checker,
        pins_thread_fini_iterators_checker
    },
    { NULL }
};

static void pins_thread_init_iterators_checker(struct parsec_execution_stream_s* es)
{
    struct parsec_pins_next_callback_s* event_cb =
        (struct parsec_pins_next_callback_s*)malloc(sizeof(struct parsec_pins_next_callback_s));
    PINS_REGISTER(es, EXEC_BEGIN, iterators_checker_exec_count_begin, event_cb);
}

static void pins_thread_fini_iterators_checker(struct parsec_execution_stream_s* es)
{
    struct parsec_pins_next_callback_s* event_cb;
    PINS_UNREGISTER(es, EXEC_BEGIN, iterators_checker_exec_count_begin, &event_cb);
    free(event_cb);
}

/*
 PINS CALLBACKS
 */

#define TASK_STR_LEN 256

static parsec_ontask_iterate_t print_link(parsec_execution_stream_t *es,
                                          const parsec_task_t *newcontext,
                                          const parsec_task_t *oldcontext,
                                          const dep_t* dep,
                                          parsec_dep_data_description_t* data,
                                          int src_rank, int dst_rank, int dst_vpid,
                                          void *param)
{
    char  new_str[TASK_STR_LEN];
    char  old_str[TASK_STR_LEN];
    char *info = (char*)param;

    parsec_task_snprintf(old_str, TASK_STR_LEN, oldcontext);
    parsec_task_snprintf(new_str, TASK_STR_LEN, newcontext);

    fprintf(stderr, "PINS ITERATORS CHECKER::   %s that runs on rank %d, vpid %d is a %s of %s that runs on rank %d.\n",
            new_str, dst_rank, dst_vpid, info, old_str, src_rank);

    (void)es; (void)dep; (void)data;
    return PARSEC_ITERATE_CONTINUE;
}

static void
iterators_checker_exec_count_begin(parsec_execution_stream_t* es,
                                   parsec_task_t* task,
                                   struct parsec_pins_next_callback_s* _data)
{
    char  str[TASK_STR_LEN];
    const dep_t *final_deps[MAX_PARAM_COUNT];
    parsec_data_t *data;
    int nbfo, i;

    parsec_task_snprintf(str, TASK_STR_LEN, task);

    if( task->task_class->iterate_successors )
        task->task_class->iterate_successors(es, task, PARSEC_DEPENDENCIES_BITMASK, print_link, "successor");
    else
        fprintf(stderr, "PINS ITERATORS CHECKER::   %s has no successor\n", str);

    if( task->task_class->iterate_predecessors )
        task->task_class->iterate_predecessors(es, task, PARSEC_DEPENDENCIES_BITMASK, print_link, "predecessor");
    else
        fprintf(stderr, "PINS ITERATORS CHECKER::   %s has no predecessor\n", str);

    nbfo = parsec_task_deps_with_final_output(task, final_deps);
    fprintf(stderr, "PINS ITERATORS CHECKER::   %s does %d final outputs.\n",
            str, nbfo);
    for(i = 0; i < nbfo; i++) {
        data = final_deps[i]->direct_data(task->taskpool, task->locals);
        if( NULL != data )
            fprintf(stderr, "PINS ITERATORS CHECKER::   %s final output number %d/%d key is %u, on device %d. \n",
                    str, i, nbfo, data->key, data->owner_device);
        else
            fprintf(stderr, "PINS ITERATORS CHECKER::   %s final output number %d/%d is remote\n",
                    str, i, nbfo);
    }
    (void)_data;
}
