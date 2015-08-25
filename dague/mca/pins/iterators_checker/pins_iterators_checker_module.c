/*
 * Copyright (c) 2013-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague/mca/pins/pins.h"
#include "pins_iterators_checker.h"
#include "dague/profiling.h"
#include "dague/execution_unit.h"
#include "dague/data_internal.h"

#include <errno.h>
#include <stdio.h>

/* init functions */
static void pins_thread_init_iterators_checker(struct dague_execution_unit_s* exec_unit);
static void pins_thread_fini_iterators_checker(struct dague_execution_unit_s* exec_unit);

/* PINS callbacks */
static void iterators_checker_exec_count_begin(dague_execution_unit_t* exec_unit,
                                               dague_execution_context_t* exec_context,
                                               struct parsec_pins_next_callback_s* data);
const dague_pins_module_t dague_pins_iterators_checker_module = {
    &dague_pins_iterators_checker_component,
    {
        NULL,
        NULL,
        NULL,
        NULL,
        pins_thread_init_iterators_checker,
        pins_thread_fini_iterators_checker
    }
};

static void pins_thread_init_iterators_checker(struct dague_execution_unit_s* exec_unit)
{
    struct parsec_pins_next_callback_s* event_cb =
        (struct parsec_pins_next_callback_s*)malloc(sizeof(struct parsec_pins_next_callback_s));
    PINS_REGISTER(exec_unit, EXEC_BEGIN, iterators_checker_exec_count_begin, event_cb);
}

static void pins_thread_fini_iterators_checker(struct dague_execution_unit_s* exec_unit)
{
    struct parsec_pins_next_callback_s* event_cb;
    PINS_UNREGISTER(exec_unit, EXEC_BEGIN, iterators_checker_exec_count_begin, &event_cb);
    free(event_cb);
}

/*
 PINS CALLBACKS
 */

#define TASK_STR_LEN 256

static dague_ontask_iterate_t print_link(dague_execution_unit_t *eu,
                                         const dague_execution_context_t *newcontext,
                                         const dague_execution_context_t *oldcontext,
                                         const dep_t* dep,
                                         dague_dep_data_description_t* data,
                                         int src_rank, int dst_rank, int dst_vpid,
                                         void *param)
{
    char  new_str[TASK_STR_LEN];
    char  old_str[TASK_STR_LEN];
    char *info = (char*)param;

    dague_snprintf_execution_context(old_str, TASK_STR_LEN, oldcontext);
    dague_snprintf_execution_context(new_str, TASK_STR_LEN, newcontext);

    fprintf(stderr, "PINS ITERATORS CHECKER::   %s that runs on rank %d, vpid %d is a %s of %s that runs on rank %d.\n",
            new_str, dst_rank, dst_vpid, info, old_str, src_rank);

    (void)eu; (void)dep; (void)data;
    return DAGUE_ITERATE_CONTINUE;
}

static void iterators_checker_exec_count_begin(dague_execution_unit_t* exec_unit,
                                               dague_execution_context_t* exec_context,
                                               struct parsec_pins_next_callback_s* _data)
{
    char  str[TASK_STR_LEN];
    const dep_t *final_deps[MAX_PARAM_COUNT];
    dague_data_t *data;
    int nbfo, i;

    dague_snprintf_execution_context(str, TASK_STR_LEN, exec_context);

    if( exec_context->function->iterate_successors )
        exec_context->function->iterate_successors(exec_unit, exec_context, DAGUE_DEPENDENCIES_BITMASK, print_link, "successor");
    else
        fprintf(stderr, "PINS ITERATORS CHECKER::   %s has no successor\n", str);

    if( exec_context->function->iterate_predecessors )
        exec_context->function->iterate_predecessors(exec_unit, exec_context, DAGUE_DEPENDENCIES_BITMASK, print_link, "predecessor");
    else
        fprintf(stderr, "PINS ITERATORS CHECKER::   %s has no predecessor\n", str);

    nbfo = dague_task_deps_with_final_output(exec_context, final_deps);
    fprintf(stderr, "PINS ITERATORS CHECKER::   %s does %d final outputs.\n",
            str, nbfo);
    for(i = 0; i < nbfo; i++) {
        data = final_deps[i]->direct_data(exec_context->dague_handle, exec_context->locals);
        if( NULL != data )
            fprintf(stderr, "PINS ITERATORS CHECKER::   %s final output number %d/%d key is %u, on device %d. \n",
                    str, i, nbfo, data->key, data->owner_device);
        else
            fprintf(stderr, "PINS ITERATORS CHECKER::   %s final output number %d/%d is remote\n",
                    str, i, nbfo);
    }
    (void)_data;
}
