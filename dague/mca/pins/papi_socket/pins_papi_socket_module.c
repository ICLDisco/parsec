/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "pins_papi_socket.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "dague/utils/output.h"
#include "dague/utils/mca_param.h"
#include <stdio.h>
#include <papi.h>
#include "execution_unit.h"

static void stop_papi_socket(dague_execution_unit_t* exec_unit,
                             dague_execution_context_t* exec_context,
                             parsec_pins_next_callback_t* cb_data);

static char* mca_param_string;
static parsec_pins_papi_events_t* pins_papi_socket_events = NULL;

static void pins_init_papi_socket(dague_context_t * master_context)
{
    pins_papi_init(master_context);

    dague_mca_param_reg_string_name("pins", "socket_event",
                                    "PAPI event to be gathered at the socket level.\n",
                                    false, false,
                                    "", &mca_param_string);
    if( NULL != mca_param_string ) {
        pins_papi_socket_events = parsec_pins_papi_events_new(mca_param_string);
    }
}

static void pins_fini_papi_socket(dague_context_t * master_context)
{
    if( NULL != pins_papi_socket_events ) {
        parsec_pins_papi_events_free(&pins_papi_socket_events);
        pins_papi_socket_events = NULL;
    }
}

static void pins_thread_init_papi_socket(dague_execution_unit_t * exec_unit)
{
    parsec_pins_papi_callback_t* event_cb = NULL;
    parsec_pins_papi_event_t* event;
    parsec_pins_papi_values_t info;
    int i, my_socket, my_core, err;
    bool started = false;
    char* conv_string = NULL;

    if( NULL == pins_papi_socket_events )
        return;

    /* Set all the matching events */
    for( i = 0; i < pins_papi_socket_events->num_counters; i++ ) {
        event = &pins_papi_socket_events->events[i];
        if( (event->socket != -1) && (event->socket != my_socket) )
            continue;
        if( (event->core != -1) && (event->core != my_core) )
            continue;

        if(!started) {  /* create the event and the PAPI eventset */
            pins_papi_thread_init(exec_unit);

            event_cb = (parsec_pins_papi_callback_t*)malloc(sizeof(parsec_pins_papi_callback_t));
            event_cb->papi_eventset = PAPI_NULL;
            event_cb->num_counters = 0;
            event_cb->events_list = pins_papi_socket_events;
            event_cb->frequency = event->frequency;
            event_cb->begin_end = 0;
            event_cb->num_tasks = 0;
            /* Create an empty eventset */
            if( PAPI_OK != (err = PAPI_create_eventset(&event_cb->papi_eventset)) ) {
                dague_output(0, "%s: thread %d couldn't create the PAPI event set; ERROR: %s\n",
                             __func__, exec_unit->th_id, PAPI_strerror(err));
                parsec_pins_papi_event_cleanup(event_cb, &info);
                free(event_cb); event_cb = NULL;
                continue;
            }
            started = true;
        }

        /* Add events to the eventset */
        if( PAPI_OK != (err = PAPI_add_event(event_cb->papi_eventset,
                                             event->pins_papi_native_event)) ) {
            dague_output(0, "%s: failed to add event %s; ERROR: %s\n",
                         __func__, event->pins_papi_event_name, PAPI_strerror(err));
            continue;
        }
        event_cb->num_counters++;
        if( NULL == conv_string )
            asprintf(&conv_string, "%s{int64_t}"PARSEC_PINS_SEPARATOR, event->pins_papi_event_name);
        else {
            char* tmp = conv_string;
            asprintf(&conv_string, "%s%s{int64_t}"PARSEC_PINS_SEPARATOR, tmp, event->pins_papi_event_name);
            free(tmp);
        }
    }

    if( (NULL != event_cb) && (0 != event_cb->num_counters) ) {
        char* key_string;

        asprintf(&key_string, "PINS_SOCKET_S%d_C%d", exec_unit->socket_id, exec_unit->core_id);

        dague_profiling_add_dictionary_keyword(key_string, "fill:#00AAFF",
                                               sizeof(long long) * event_cb->num_counters,
                                               conv_string,
                                               &event_cb->pins_prof_event[0],
                                               &event_cb->pins_prof_event[1]);
        free(key_string);

        if( PAPI_OK != (err = PAPI_start(event_cb->papi_eventset)) ) {
            dague_output(0, "couldn't start PAPI eventset for thread %d; ERROR: %s\n",
                         exec_unit->th_id, PAPI_strerror(err));
            goto cleanup_and_return;
        }

        if( PAPI_OK != (err = PAPI_read(event_cb->papi_eventset, info.values)) ) {
            dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                         exec_unit->th_id, PAPI_strerror(err));
            goto cleanup_and_return;
        }

        (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_event[event_cb->begin_end],
                                    45, 0, (void *)&info);

        event_cb->begin_end = (event_cb->begin_end + 1) & 0x1;  /* aka. % 2 */

        PINS_REGISTER(exec_unit, EXEC_END, stop_papi_socket,
                      (parsec_pins_next_callback_t*)event_cb);
        free(conv_string);
        return;  /* we're done here */
    }
  cleanup_and_return:
    if( NULL != event_cb ) {
        parsec_pins_papi_event_cleanup(event_cb, &info);
        free(event_cb); event_cb = NULL;
    }
    if( NULL != conv_string )
        free(conv_string);
}

static void pins_thread_fini_papi_socket(dague_execution_unit_t* exec_unit)
{
    parsec_pins_papi_callback_t* event_cb;
    parsec_pins_papi_values_t info;
    int err, i;

    PINS_UNREGISTER(exec_unit, EXEC_END, stop_papi_socket, (parsec_pins_next_callback_t**)&event_cb);

    if( NULL == event_cb )
        return;

    pins_papi_thread_fini(exec_unit);
    if( PAPI_NULL == event_cb->papi_eventset ) {
        parsec_pins_papi_event_cleanup(event_cb, &info);

        /* If the last profiling event was an 'end' event */
        if(event_cb->begin_end == 0) {
            (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_event[0],
                                        45, 0, (void *)&info);
        }
        (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_event[1],
                                    45, 0, (void *)&info);
    }
    free(event_cb);
}

static void stop_papi_socket(dague_execution_unit_t* exec_unit,
                             dague_execution_context_t* exec_context,
                             parsec_pins_next_callback_t* cb_data)
{
    parsec_pins_papi_callback_t* event_cb = (parsec_pins_papi_callback_t*)cb_data;
    parsec_pins_papi_values_t info;
    int err;

    event_cb->num_tasks++;
    if(event_cb->num_tasks < event_cb->frequency)
        return;

    if( PAPI_OK != (err = PAPI_read(event_cb->papi_eventset, info.values)) ) {
        dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        return;
    }
    (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_event[event_cb->begin_end],
                                45, 0, (void *)&info);
    event_cb->begin_end = (event_cb->begin_end + 1) & 0x1;  /* aka. % 2 */
    event_cb->num_tasks = 0;
}

const dague_pins_module_t dague_pins_papi_socket_module = {
    &dague_pins_papi_socket_component,
    {
        pins_init_papi_socket,
        pins_fini_papi_socket,
        NULL,
        NULL,
        pins_thread_init_papi_socket,
        pins_thread_fini_papi_socket,
    }
};
