/*
 * Copyright (c) 2012-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "pins_papi_select.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include <papi.h>
#include "debug.h"
#include "execution_unit.h"


typedef struct parsec_pins_select_callback_s {
    parsec_pins_next_callback_t     default_cb;
    int                             papi_eventset;
} parsec_pins_select_callback_t;

static int select_events[NUM_SELECT_EVENTS] = {PAPI_L2_TCM, PAPI_L2_DCM};

#define THREAD_NUM(exec_unit) (exec_unit->virtual_process->vp_id *      \
                               exec_unit->virtual_process->dague_context->nb_vp + \
                               exec_unit->th_id )

static void pins_init_papi_select(dague_context_t * master_context);
static void pins_thread_init_papi_select(dague_execution_unit_t * exec_unit);
static void pins_thread_fini_papi_select(dague_execution_unit_t * exec_unit);

const dague_pins_module_t dague_pins_papi_select_module = {
    &dague_pins_papi_select_component,
    {
        pins_init_papi_select,
        NULL,
        NULL,
        NULL,
        pins_thread_init_papi_select,
        pins_thread_fini_papi_select
    }
};

static void start_papi_select_count(dague_execution_unit_t * exec_unit,
                                    dague_execution_context_t * exec_context,
                                    parsec_pins_next_callback_t* cb_data);
static void stop_papi_select_count(dague_execution_unit_t * exec_unit,
                                   dague_execution_context_t * exec_context,
                                   parsec_pins_next_callback_t* cb_data);

static int pins_prof_select_begin, pins_prof_select_end;
static int papi_select_enabled = 0;

static void pins_init_papi_select(dague_context_t * master_context)
{
    pins_papi_init(master_context);

    dague_profiling_add_dictionary_keyword("PINS_SELECT", "fill:#0000FF",
                                           sizeof(select_info_t), NULL,
                                           &pins_prof_select_begin, &pins_prof_select_end);

    papi_select_enabled = parsec_pins_is_module_enabled("papi_select");
}

static void pins_thread_init_papi_select(dague_execution_unit_t * exec_unit)
{
    parsec_pins_select_callback_t* event_cb;
    int rv;

    pins_papi_thread_init(exec_unit);

    event_cb = (parsec_pins_select_callback_t*)malloc(sizeof(parsec_pins_select_callback_t));
    if( NULL == event_cb ) {
        DEBUG(("PINS thread init failed to allocate memory."));
        return;
    }

    event_cb->papi_eventset = PAPI_NULL;
    if (PAPI_create_eventset(&event_cb->papi_eventset) != PAPI_OK) {
        DEBUG(("papi_select.c, pins_thread_init_papi_select: "
               "failed to create SELECT event set\n"));
        return;
    }
    if ((rv = PAPI_add_events(event_cb->papi_eventset,
                              select_events, NUM_SELECT_EVENTS)) != PAPI_OK) {
        DEBUG(("papi_select.c, pins_thread_init_papi_select: failed to add "
               "steal events to StealEventSet. %d %s\n", rv, PAPI_strerror(rv)));
        return;
    }

    PINS_REGISTER(exec_unit, SELECT_BEGIN, start_papi_select_count, (parsec_pins_next_callback_t*)event_cb);
    PINS_REGISTER(exec_unit, SELECT_END,    stop_papi_select_count, (parsec_pins_next_callback_t*)event_cb);
}

static void pins_thread_fini_papi_select(dague_execution_unit_t * exec_unit)
{
    parsec_pins_select_callback_t* event_cb;

    PINS_UNREGISTER(exec_unit, SELECT_BEGIN, start_papi_select_count,
                    (parsec_pins_next_callback_t**)&event_cb);
    free(event_cb);
    PINS_UNREGISTER(exec_unit, SELECT_END,    stop_papi_select_count,
                    (parsec_pins_next_callback_t**)&event_cb);
    free(event_cb);
}

static void start_papi_select_count(dague_execution_unit_t * exec_unit,
                                    dague_execution_context_t * exec_context,
                                    parsec_pins_next_callback_t* cb_data)
{
    parsec_pins_select_callback_t* event_cb = (parsec_pins_select_callback_t*)cb_data;
    int rv;

    if ((rv = PAPI_start(event_cb->papi_eventset)) != PAPI_OK) {
        DEBUG(("%p papi_select.c, start_papi_select_count: "
               "can't start SELECT event counters! %d %s\n",
               exec_unit, rv, PAPI_strerror(rv)));
        return;
    }
    dague_profiling_trace(exec_unit->eu_profile,
                          pins_prof_select_begin,
                          45,
                          0,
                          NULL);
    (void)exec_unit;
}

static void stop_papi_select_count(dague_execution_unit_t * exec_unit,
                                   dague_execution_context_t * exec_context,
                                   parsec_pins_next_callback_t* cb_data)
{
    parsec_pins_select_callback_t* event_cb = (parsec_pins_select_callback_t*)cb_data;
    unsigned int num_threads = (exec_unit->virtual_process->dague_context->nb_vp
                                * exec_unit->virtual_process->nb_cores);
    select_info_t info;
    int rv = PAPI_OK;

    info.kernel_type = -1;

    if (exec_context) {
        if (exec_context->dague_handle->profiling_array != NULL)
            info.kernel_type = exec_context->dague_handle->profiling_array[exec_context->function->function_id * 2] / 2;
    }
    info.victim_vp_id = -1; // currently unavailable from scheduler queue object
    /* TODO: this simply cannot work. */
    unsigned long long victim_core_num = 0;
    if (victim_core_num >= num_threads)
        info.victim_vp_id = SYSTEM_QUEUE_VP;
    info.victim_th_id = (int)victim_core_num; // but this number includes the vp id multiplier
    info.exec_context = (unsigned long long int)exec_context; // if NULL, this was starvation

    // now count the PAPI events, if available
    if ((rv = PAPI_stop(event_cb->papi_eventset, info.values)) != PAPI_OK) {
        DEBUG(("papi_select.c, stop_papi_select_count: "
               "can't stop SELECT event counters! %d %s\n",
               rv, PAPI_strerror(rv)));
        for (int i = 0; i < NUM_SELECT_EVENTS; info.values[i] = -1, i++);  // default values
    }

    dague_profiling_trace(exec_unit->eu_profile,
                          pins_prof_select_end,
                          45,
                          0,
                          (void *)&info);
}
