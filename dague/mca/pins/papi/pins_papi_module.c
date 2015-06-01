/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "pins_papi.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "dague/utils/output.h"
#include "dague/utils/mca_param.h"
#include <stdio.h>
#include <papi.h>
#include "dague/execution_unit.h"

static void pins_papi_read(dague_execution_unit_t* exec_unit,
                             dague_execution_context_t* exec_context,
                             parsec_pins_next_callback_t* cb_data);

static char* mca_param_string;
static parsec_pins_papi_events_t* pins_papi_events = NULL;

static void pins_init_papi(dague_context_t * master_context)
{
    pins_papi_init(master_context);

    dague_mca_param_reg_string_name("pins", "papi_event",
                                    "PAPI events to be gathered at both socket and core level.\n",
                                    false, false,
                                    "", &mca_param_string);
    if( NULL != mca_param_string ) {
        pins_papi_events = parsec_pins_papi_events_new(mca_param_string);
    }
}

static void pins_fini_papi(dague_context_t * master_context)
{
    if( NULL != pins_papi_events ) {
        parsec_pins_papi_events_free(&pins_papi_events);
        pins_papi_events = NULL;
    }
}

static int register_event_cb(dague_execution_unit_t * exec_unit,
                             parsec_pins_papi_callback_t* event_cb,
                             const char* conv_string,
                             int event_id)
{
    char* key_string;
    parsec_pins_papi_values_t info;
    int err;

    assert( NULL != event_cb );
    asprintf(&key_string, "PINS_PAPI_S%d_C%d_F%d_%d",
             exec_unit->socket_id, exec_unit->core_id, event_cb->frequency, event_id);

    dague_profiling_add_dictionary_keyword(key_string, "fill:#00AAFF",
                                           sizeof(long long) * event_cb->num_counters,
                                           conv_string,
                                           &event_cb->pins_prof_event[0],
                                           &event_cb->pins_prof_event[1]);
    free(key_string);

    if( PAPI_OK != (err = PAPI_start(event_cb->papi_eventset)) ) {
        dague_output(0, "couldn't start PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        event_cb->num_counters = 0;
        return DAGUE_ERROR;
    }

    if( PAPI_OK != (err = PAPI_read(event_cb->papi_eventset, info.values)) ) {
        dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        return DAGUE_ERROR;
    }
    dague_output(0, "PAPI event %s core %d socket %d frequency %d enabled\n",
                 conv_string, event_cb->event->core, event_cb->event->socket, event_cb->event->frequency);

    (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_event[event_cb->begin_end],
                                45, 0, (void *)&info);

    event_cb->begin_end = (event_cb->begin_end + 1) & 0x1;  /* aka. % 2 */

    if(event_cb->frequency == 1) {
        PINS_REGISTER(exec_unit, EXEC_BEGIN, pins_papi_read,
                      (parsec_pins_next_callback_t*)event_cb);
    }
    PINS_REGISTER(exec_unit, EXEC_END, pins_papi_read,
                  (parsec_pins_next_callback_t*)event_cb);
    return DAGUE_SUCCESS;
}

static void pins_thread_init_papi(dague_execution_unit_t * exec_unit)
{
    parsec_pins_papi_callback_t* event_cb = NULL;
    parsec_pins_papi_event_t* event;
    parsec_pins_papi_values_t info;
    int i, my_socket, my_core, err, event_id = 0;
    char *conv_string = NULL, *datatype;

    if( NULL == pins_papi_events )
        return;

    my_socket = exec_unit->socket_id;
    my_core = exec_unit->core_id;

    pins_papi_thread_init(exec_unit);

    for( i = 0; i < pins_papi_events->num_counters; i++ ) {
        event = pins_papi_events->events[i];
        conv_string = NULL;

        for( ; NULL != event; event = event->next ) {
            if( (event->socket != -1) && (event->socket != my_socket) )
                continue;
            if( (event->core != -1) && (event->core != my_core) )
                continue;

            if( NULL == event_cb ) {  /* create the event and the PAPI eventset */

              force_a_new_event_cb:  /* read the comment below to understand why we need this label */
                event_cb = (parsec_pins_papi_callback_t*)malloc(sizeof(parsec_pins_papi_callback_t));

                event_cb->papi_eventset = PAPI_NULL;
                event_cb->num_counters  = 0;
                event_cb->event         = event;
                event_cb->frequency     = event->frequency;
                event_cb->begin_end     = 0;

                /* Create an empty eventset */
                if( PAPI_OK != (err = PAPI_create_eventset(&event_cb->papi_eventset)) ) {
                    dague_output(0, "%s: thread %d couldn't create the PAPI event set; ERROR: %s\n",
                                 __func__, exec_unit->th_id, PAPI_strerror(err));
                    /* Destroy the event it is unsafe to use */
                    free(event_cb); event_cb = NULL;
                    continue;
                }
            } else {
                /* We are trying to hook on an already allocated event. If we don't have the same frequency
                 * we should fail and force the registration of the event_cb built so far, and then finally
                 * add this event again to an empty event_cb.
                 */
                if( event_cb->frequency != event->frequency ) {

                    err = register_event_cb(exec_unit, event_cb, conv_string, event_id);
                    event_id++;  /* next event_id */
                    free(conv_string);
                    conv_string = NULL;
                    if( DAGUE_SUCCESS == err )
                        /* We registered the event_cb built so far. We now need to create a fresh one for the ongoing event. */
                        goto force_a_new_event_cb;

                    /* We have a failure. Destroy the event_cb and move forward */
                    parsec_pins_papi_event_cleanup(event_cb, &info);
                    free(event_cb); event_cb = NULL;
                }
            }
            /* Add events to the eventset */
            if( PAPI_OK != (err = PAPI_add_event(event_cb->papi_eventset,
                                                 event->pins_papi_native_event)) ) {
                dague_output(0, "%s: failed to add event %s; ERROR: %s\n",
                             __func__, event->pins_papi_event_name, PAPI_strerror(err));
                continue;
            }
            event_cb->num_counters++;
            switch( event->papi_data_type ) {
            case PAPI_DATATYPE_INT64: datatype = "int64_t"; break;
            case PAPI_DATATYPE_UINT64: datatype = "uint64_t"; break;
            case PAPI_DATATYPE_FP64: datatype = "double"; break;
            case PAPI_DATATYPE_BIT64: datatype = "int64_t"; break;
            default: datatype = "int64_t"; break;
            }

            if( NULL == conv_string )
                asprintf(&conv_string, "%s{%s}"PARSEC_PINS_SEPARATOR, event->pins_papi_event_name, datatype);
            else {
                char* tmp = conv_string;
                asprintf(&conv_string, "%s%s{%s}"PARSEC_PINS_SEPARATOR, tmp, event->pins_papi_event_name, datatype);
                free(tmp);
            }
        }
    }

    if( NULL != event_cb ) {
        if( DAGUE_SUCCESS != (err = register_event_cb(exec_unit, event_cb, conv_string, event_id)) ) {
            parsec_pins_papi_event_cleanup(event_cb, &info);
            free(event_cb);
        }
    }
    if( NULL != conv_string )
        free(conv_string);
}

static void pins_thread_fini_papi(dague_execution_unit_t* exec_unit)
{
    parsec_pins_papi_callback_t* event_cb;
    parsec_pins_papi_values_t info;
    int err, i;

    do {
        /* Should this be in a loop to unregister all the callbacks for this thread? */
        PINS_UNREGISTER(exec_unit, EXEC_END, pins_papi_read, (parsec_pins_next_callback_t**)&event_cb);
        if( NULL == event_cb )
            return;

        if( 1 == event_cb->frequency ) {  /* this must have an EXEC_BEGIN */
            parsec_pins_papi_callback_t* start_cb;
            PINS_UNREGISTER(exec_unit, EXEC_BEGIN, pins_papi_read, (parsec_pins_next_callback_t**)&start_cb);
            if( NULL == start_cb ) {
                dague_output(0, "Unsettling exception of an event with frequency 1 but without a start callback. Ignored.\n");
            }
        }
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
    } while(1);

    pins_papi_thread_fini(exec_unit);
}

static void pins_papi_read(dague_execution_unit_t* exec_unit,
                           dague_execution_context_t* exec_context,
                           parsec_pins_next_callback_t* cb_data)
{
    parsec_pins_papi_callback_t* event_cb = (parsec_pins_papi_callback_t*)cb_data;
    parsec_pins_papi_values_t info;
    int i, err;

    if(1 == event_cb->frequency ) {  /* trigger the event */

        if( PAPI_OK != (err = PAPI_read(event_cb->papi_eventset, info.values)) ) {
            dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                         exec_unit->th_id, PAPI_strerror(err));
            return;
        }
        (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_event[event_cb->begin_end],
                                    45, 0, (void *)&info);
        event_cb->begin_end = (event_cb->begin_end + 1) & 0x1;  /* aka. % 2 */
        event_cb->frequency = event_cb->event->frequency;
    } else event_cb->frequency--;
}

const dague_pins_module_t dague_pins_papi_module = {
    &dague_pins_papi_component,
    {
        pins_init_papi,
        pins_fini_papi,
        NULL,
        NULL,
        pins_thread_init_papi,
        pins_thread_fini_papi,
    }
};
