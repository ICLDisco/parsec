/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* PaRSEC Performance Instrumentation Callback System */
#include "parsec/parsec_config.h"
#include <stdlib.h>
#include <assert.h>
#include "parsec/mca/pins/pins.h"
#include "parsec/debug.h"
#include "parsec/execution_stream.h"

static int registration_disabled;

void parsec_pins_instrument(struct parsec_execution_stream_s* es,
                            PINS_FLAG method_flag,
                            parsec_task_t* task)
{
    assert( method_flag < PINS_FLAG_COUNT );

    parsec_pins_next_callback_t* cb_event = &es->pins_events_cb[method_flag];
    while( NULL != cb_event->cb_func ) {
        cb_event->cb_func(es, task, cb_event->cb_data);
        cb_event = cb_event->cb_data;
    }
}

/* convenience method provided 'just in case' */
void parsec_pins_disable_registration(int disable)
{
    parsec_debug_verbose(5, parsec_debug_output, "PINS registration is %s.",
                        disable ? "disabled" : "enabled");
    registration_disabled = disable;
}

/**
 * Register a new callback for a particular PINS event. The provided cb_data
 * must not be NULL, as it is used to chain the callback with all the previous
 * ones.
 */
int parsec_pins_register_callback(struct parsec_execution_stream_s* es,
                                  PINS_FLAG method_flag,
                                  parsec_pins_callback cb_func,
                                  struct parsec_pins_next_callback_s* cb_data)
{
    if( method_flag >= PINS_FLAG_COUNT ) {
        parsec_warning("PINS register MUST be called with a valid event flag.");
        return -1;
    }
    if( NULL == cb_data ) {
        parsec_warning("PINS registration MUST be called with non-NULL data. Discard PINS module");
        return -1;
    }
    if (registration_disabled) {
        parsec_inform("PINS has been disabled by command line argument, causing this registration to have no effect.");
        return 0;
    }

    parsec_pins_next_callback_t* cb_event = &es->pins_events_cb[method_flag];

    *cb_data = *cb_event;

    cb_event->cb_func = cb_func;
    cb_event->cb_data = cb_data;

    return 0;
}

int parsec_pins_unregister_callback(struct parsec_execution_stream_s* es,
                                    PINS_FLAG method_flag,
                                    parsec_pins_callback cb,
                                    struct parsec_pins_next_callback_s** cb_data)
{
    *cb_data = NULL;
    if( method_flag >= PINS_FLAG_COUNT ) {
        parsec_warning("PINS unregister MUST be called with a valid event flag.");
        return -1;
    }
    if (registration_disabled) {
        parsec_inform("PINS has been disabled by command line argument, causing this UN-registration to have no effect.");
        return 0;
    }

    parsec_pins_next_callback_t* cb_event = &es->pins_events_cb[method_flag];
    while( (NULL != cb_event->cb_data) && (cb != cb_event->cb_func) ) {
        cb_event = cb_event->cb_data;
    }
    if( NULL == cb_event->cb_data ) {
        /* No matching event could be found in the list */
        return -1;
    }
    assert(cb_event->cb_func == cb);
    *cb_data = cb_event->cb_data;
    *cb_event = **cb_data;
    return 0;
}
