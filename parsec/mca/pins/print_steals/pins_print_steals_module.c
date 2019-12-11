/*
 * Copyright (c) 2012-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "pins_print_steals.h"
#include "parsec/mca/pins/pins.h"
#ifdef PARSEC_HAVE_PAPI
#include <papi.h>
#endif
#include "parsec/utils/debug.h"
#include "parsec/execution_stream.h"

static void pins_init_print_steals(parsec_context_t* master_context);
static void pins_thread_init_print_steals(parsec_execution_stream_t* es);
static void pins_thread_fini_print_steals(parsec_execution_stream_t* es);

const parsec_pins_module_t parsec_pins_print_steals_module = {
    &parsec_pins_print_steals_component,
    {
        pins_init_print_steals,
        NULL,
        NULL,
        NULL,
        pins_thread_init_print_steals,
        pins_thread_fini_print_steals
    },
    { NULL }
};

typedef struct parsec_pins_print_steals_data_s {
    parsec_pins_next_callback_t cb_data;
    long steal_counters[1];
} parsec_pins_print_steals_data_t;

static void stop_print_steals_count(parsec_execution_stream_t* es,
                                    parsec_task_t* task,
                                    parsec_pins_next_callback_t* data);

#define THREAD_NUM(exec_unit) (exec_unit->virtual_process->vp_id *      \
                               exec_unit->virtual_process->parsec_context->nb_vp + \
                               exec_unit->th_id )

static int total_cores;

static void pins_init_print_steals(parsec_context_t* master)
{
    total_cores = master->nb_vp * master->virtual_processes[0]->nb_cores;
}

static void pins_thread_init_print_steals(parsec_execution_stream_t* es)
{
    parsec_pins_print_steals_data_t* event_cb =
        (parsec_pins_print_steals_data_t*)calloc(1, sizeof(parsec_pins_print_steals_data_t) +
                                                 (total_cores + 2) * sizeof(long));
    PARSEC_PINS_REGISTER(es, SELECT_END, stop_print_steals_count,
                  (parsec_pins_next_callback_t*)event_cb);
}

static void pins_thread_fini_print_steals(parsec_execution_stream_t* es)
{
    parsec_pins_print_steals_data_t* event_cb;
    PARSEC_PINS_UNREGISTER(es, SELECT_END, stop_print_steals_count,
                  (parsec_pins_next_callback_t**)&event_cb);

    for (int k = 0; k < total_cores + 2; k++)
        printf("%7ld ", event_cb->steal_counters[k]);
    printf("\n");
    free(event_cb);
}

static void stop_print_steals_count(parsec_execution_stream_t* es,
                                    parsec_task_t* task,
                                    parsec_pins_next_callback_t* data)
{
    parsec_pins_print_steals_data_t* event_cb = (parsec_pins_print_steals_data_t*)data;
    /**
     * This is plain wrong and it could not have been working.
     */
    unsigned long long victim_core_num = 0;

    if (task != NULL)
        event_cb->steal_counters[victim_core_num] += 1;
    else
        event_cb->steal_counters[victim_core_num + 1] += 1;
    (void)es;
}

