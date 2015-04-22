/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "pins_papi_exec.h"
#include <papi.h>
#include <stdio.h>
#include "profiling.h"
#include "execution_unit.h"

/* these should eventually be runtime-configurable */
static int exec_events[NUM_EXEC_EVENTS] = PAPI_EXEC_NATIVE_EVENT_NAMES;

static void pins_init_papi_exec(dague_context_t * master_context);
static void pins_thread_init_papi_exec(dague_execution_unit_t * exec_unit);
static void pins_thread_fini_papi_exec(dague_execution_unit_t * exec_unit);

const dague_pins_module_t dague_pins_papi_exec_module = {
    &dague_pins_papi_exec_component,
    {
        pins_init_papi_exec,
        NULL,
        NULL,
        NULL,
        pins_thread_init_papi_exec,
        pins_thread_fini_papi_exec,
    }
};

static void start_papi_exec_count(dague_execution_unit_t * exec_unit,
                                  dague_execution_context_t * exec_context,
                                  parsec_pins_next_callback_t* data);
static void stop_papi_exec_count(dague_execution_unit_t * exec_unit,
                                 dague_execution_context_t * exec_context,
                                 parsec_pins_next_callback_t* data);

typedef struct parsec_pins_exec_callback_s {
    parsec_pins_next_callback_t     default_cb;
    int                             papi_eventset;
} parsec_pins_exec_callback_t;

static int papi_socket_enabled = 0;

static int pins_prof_papi_exec_begin, pins_prof_papi_exec_end;

static void pins_init_papi_exec(dague_context_t * master_context)
{
    pins_papi_init(master_context);

    dague_profiling_add_dictionary_keyword("PINS_EXEC", "fill:#00FF00",
                                           sizeof(papi_exec_info_t), "kernel_type{int32_t}:value1{int64_t}:value2{int64_t}:value3{int64_t}:value4{int64_t}",
                                           &pins_prof_papi_exec_begin, &pins_prof_papi_exec_end);

    papi_socket_enabled = parsec_pins_is_module_enabled("papi_socket");
}


static void pins_thread_init_papi_exec(dague_execution_unit_t * exec_unit)
{
    parsec_pins_exec_callback_t* event_cb;
    int rv = 0;

    pins_papi_thread_init(exec_unit);

    if(!papi_socket_enabled ||
        exec_unit->th_id % CORES_PER_SOCKET != WHICH_CORE_IN_SOCKET) {
        return;
    }

    event_cb = (parsec_pins_exec_callback_t*)malloc(sizeof(parsec_pins_exec_callback_t));
    if( NULL == event_cb ) {
        DEBUG(("PINS thread init failed to allocate memory."));
        return;
    }

    event_cb->papi_eventset = PAPI_NULL;
    if( PAPI_OK != (rv = PAPI_create_eventset(&event_cb->papi_eventset))) {
        DEBUG(("papi_exec_module.c, pins_thread_init_papi_exec: "
               "thread %d couldn't create the PAPI event set "
               "to measure L1/L2 misses; ERROR: %s\n",
               exec_unit->th_id, PAPI_strerror(rv)));
        return;
    }
    if( PAPI_OK != (rv = PAPI_add_events(event_cb->papi_eventset,
                                         exec_events, NUM_EXEC_EVENTS))) {
        DEBUG(("papi_exec.c, pins_thread_init_papi_exec: thread %d failed to add "
               "exec events to EXEC eventset; ERROR: %s\n",
               exec_unit->th_id, PAPI_strerror(rv)));
        return;
    }
    PINS_REGISTER(exec_unit, EXEC_BEGIN, start_papi_exec_count,
                  (parsec_pins_next_callback_t*)event_cb);
    PINS_REGISTER(exec_unit, EXEC_END, stop_papi_exec_count,
                  (parsec_pins_next_callback_t*)event_cb);
}

static void pins_thread_fini_papi_exec(dague_execution_unit_t * exec_unit)
{
    parsec_pins_next_callback_t* event_cb;

    PINS_UNREGISTER(exec_unit, EXEC_BEGIN, start_papi_exec_count, &event_cb);
    /* David: add the code to cleanup the PAPI events */
    free(event_cb);
    PINS_UNREGISTER(exec_unit, EXEC_END,   stop_papi_exec_count, &event_cb);
    free(event_cb);
}

static void start_papi_exec_count(dague_execution_unit_t* exec_unit,
                                  dague_execution_context_t* exec_context,
                                  parsec_pins_next_callback_t* cb_data)
{
    parsec_pins_exec_callback_t* event_cb = (parsec_pins_exec_callback_t*)cb_data;
    int rv = PAPI_OK;

    assert(!papi_socket_enabled || (exec_unit->th_id % CORES_PER_SOCKET != WHICH_CORE_IN_SOCKET));
    if ((rv = PAPI_start(event_cb->papi_eventset)) != PAPI_OK) {
        printf("papi_exec.c, start_papi_exec_count: thread %d can't start "
               "exec event counters! %s\n",
               exec_unit->th_id, PAPI_strerror(rv));
        return;
    }
    dague_profiling_trace(exec_unit->eu_profile, pins_prof_papi_exec_begin,
                          (*exec_context->function->key
                           )(exec_context->dague_handle, exec_context->locals),
                          exec_context->dague_handle->handle_id, NULL);
}

static void stop_papi_exec_count(dague_execution_unit_t * exec_unit,
                                 dague_execution_context_t * exec_context,
                                 parsec_pins_next_callback_t* cb_data)
{
    parsec_pins_exec_callback_t* event_cb = (parsec_pins_exec_callback_t*)cb_data;
    papi_exec_info_t info;
    int rv;

    info.kernel_type = -1;
    assert(!papi_socket_enabled || (exec_unit->th_id % CORES_PER_SOCKET != WHICH_CORE_IN_SOCKET));
    if ((rv = PAPI_stop(event_cb->papi_eventset, info.values)) != PAPI_OK) {
        DEBUG(("papi_exec_module.c, stop_papi_exec_count: "
               "thread %d can't stop exec event counters! "
               "ERROR: %s\n",
               exec_unit->th_id, PAPI_strerror(rv)));
        for (int i = 0; i < NUM_EXEC_EVENTS; info.values[i] = -1, i++);  // default values
    }

    if (exec_context->dague_handle->profiling_array != NULL)
        info.kernel_type = exec_context->dague_handle->profiling_array[exec_context->function->function_id * 2] / 2;

    /* not *necessary*, but perhaps better for compatibility
     * with the Python dbpreader script in the long term,
     * since this will allow the reading of different structs.
     * presumably, a 'generic' Cython info reader could be created
     * that allows a set of ints and a set of long longs
     * to be automatically read if both lengths are included,
     * e.g. struct { int num_ints; int; int; int; int num_lls;
     * ll; ll; ll; ll; ll; ll } - the names could be assigned
     * after the fact by a knowledgeable end user */
    /* info.values_len = NUM_EXEC_EVENTS; */

    rv = dague_profiling_trace(exec_unit->eu_profile, pins_prof_papi_exec_end,
                               (*exec_context->function->key )(exec_context->dague_handle, exec_context->locals),
                               exec_context->dague_handle->handle_id,
                               (void *)&info);
    if (rv) {
        DEBUG(("failed to save PINS_EXEC event to profiling system %d\n", rv));
    }
    (void)exec_context;
}
