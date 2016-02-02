/*
 * Copyright (c) 2009-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* PaRSEC Performance Instrumentation Callback System */
#include "dague_config.h"
#include <stdlib.h>
#include <assert.h>
#include "dague/mca/pins/pins.h"
#include "dague/debug.h"
#include "dague/execution_unit.h"

static int registration_disabled;

void parsec_pins_instrument(struct dague_execution_unit_s* exec_unit,
                            PINS_FLAG method_flag,
                            struct dague_execution_context_s* task)
{
    assert( method_flag < PINS_FLAG_COUNT );

    parsec_pins_next_callback_t* cb_event = &exec_unit->pins_events_cb[method_flag];
    while( NULL != cb_event->cb_func ) {
        cb_event->cb_func(exec_unit, task, cb_event->cb_data);
        cb_event = cb_event->cb_data;
    }
}

/* convenience method provided 'just in case' */
void parsec_pins_disable_registration(int disable)
{
    if (disable) {
        dague_debug_verbose(5, dague_debug_output, "PINS registration is disabled.\n");
    } else {
        dague_debug_verbose(5, dague_debug_output, "PINS registration is enabled.\n");
    }
    registration_disabled = disable;
}

/**
 * Register a new callback for a particular PINS event. The provided cb_data
 * must not be NULL, as it is used to chain the callback with all the previous
 * ones.
 */
int parsec_pins_register_callback(struct dague_execution_unit_s* exec_unit,
                                  PINS_FLAG method_flag,
                                  parsec_pins_callback cb_func,
                                  struct parsec_pins_next_callback_s* cb_data)
{
    if( method_flag >= PINS_FLAG_COUNT ) {
        dague_warning("PINS register MUST be called on a non valid type of event.");
        return -1;
    }
    if( NULL == cb_data ) {
        dague_warning("PINS registration MUST be called with a non-NULL data. Discard PINS module");
        return -1;
    }
    if (registration_disabled) {
        dague_inform("PINS has been disabled by command line argument, causing this registration to have no effect.");
        return 0;
    }

    parsec_pins_next_callback_t* cb_event = &exec_unit->pins_events_cb[method_flag];

    *cb_data = *cb_event;

    cb_event->cb_func = cb_func;
    cb_event->cb_data = cb_data;

    return 0;
}

int parsec_pins_unregister_callback(struct dague_execution_unit_s* exec_unit,
                                    PINS_FLAG method_flag,
                                    parsec_pins_callback cb,
                                    struct parsec_pins_next_callback_s** cb_data)
{
    *cb_data = NULL;
    if( method_flag >= PINS_FLAG_COUNT ) {
        dague_warning("PINS unregister MUST called on a non valid type of event.");
        return -1;
    }
    if (registration_disabled) {
        dague_inform("PINS has been disabled by command line argument, causing this UN-registration to have no effect.");
        return 0;
    }

    parsec_pins_next_callback_t* cb_event = &exec_unit->pins_events_cb[method_flag];
    while( (NULL != cb_event->cb_data) && (cb != cb_event->cb_func) ) {
        cb_event = cb_event->cb_data;
    }
    if( NULL == cb_event->cb_data ) {
        dague_debug_verbose(3, dague_debug_output, "Unmatched call to PINS unregister");
        return -1;
    }
    assert(cb_event->cb_func == cb);
    *cb_data = cb_event->cb_data;
    *cb_event = **cb_data;
    return 0;
}
