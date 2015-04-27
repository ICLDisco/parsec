/*
 * Copyright (c) 2009-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* PaRSEC Performance Instrumentation Callback System */
#include "dague_config.h"
#include <stdlib.h>
#include <assert.h>
#include "pins.h"
#include "debug.h"
#include "execution_unit.h"

static int registration_disabled;

parsec_pins_callback * pins_array[PINS_FLAG_COUNT] = { 0 };
//parsec_pins_callback * pins_callback_array[PINS_FLAG_COUNT] = {0};

void parsec_pins_instrument(struct dague_execution_unit_s* exec_unit,
                            PINS_FLAG method_flag,
                            struct dague_execution_context_s* task)
{
    assert( (method_flag >= 0) && (method_flag < PINS_FLAG_COUNT) );

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
        DEBUG3(("PINS registration is disabled.\n"));
    } else {
        DEBUG3(("PINS registration is enabled.\n"));
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
    if( (method_flag < 0) || (method_flag >= PINS_FLAG_COUNT) ) {
        DEBUG(("PINS register MUST called on a non valid type of event."));
        return -1;
    }
    if( NULL == cb_data ) {
        DEBUG(("PINS registration MUST be called with a non-NULL data. Discard PINS module"));
        return -1;
    }
    if (registration_disabled) {
        DEBUG2(("NOTE: PINS has been disabled by command line argument, causing this registration to be ignored."));
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
    if( (method_flag < 0) || (method_flag >= PINS_FLAG_COUNT) ) {
        DEBUG(("PINS unregister MUST called on a non valid type of event."));
        return -1;
    }
    if (registration_disabled) {
        DEBUG3(("NOTE: PINS has been disabled by command line argument, causing this UN-registration to fail."));
        return -1;
    }

    parsec_pins_next_callback_t* cb_event = &exec_unit->pins_events_cb[method_flag];
    while( (NULL != cb_event->cb_data) && (cb != cb_event->cb_func) ) {
        cb_event = cb_event->cb_data;
    }
    if( NULL == cb_event->cb_data ) {
        DEBUG(("Unmatched call to PINS unregister"));
        return -1;
    }
    assert(cb_event->cb_func == cb);
    *cb_data = cb_event->cb_data;
    *cb_event = **cb_data;
    return 0;
}
