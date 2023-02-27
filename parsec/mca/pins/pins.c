/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

/* PaRSEC Performance Instrumentation Callback System */
#include "parsec/parsec_config.h"
#include <stdlib.h>
#include <assert.h>
#include "parsec/mca/pins/pins.h"
#include "parsec/constants.h"
#include "parsec/utils/debug.h"
#include "parsec/execution_stream.h"

/**
 * Mask for PINS events that are enabled by default.
 * Note that we only use BEGIN events here,
 * END events map to the same mask element.
 * We do not enable BEGIN and END events separately.
 */
uint64_t parsec_pins_enable_mask = PARSEC_PINS_FLAG_MASK(RELEASE_DEPS_BEGIN)
                                 | PARSEC_PINS_FLAG_MASK(EXEC_BEGIN);

const char *parsec_pins_enable_default_names = "release_deps,exec_begin";

static int registration_disabled;

void parsec_pins_instrument(struct parsec_execution_stream_s* es,
                            PARSEC_PINS_FLAG method_flag,
                            parsec_task_t* task)
{
    assert( method_flag < PARSEC_PINS_FLAG_COUNT );

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
                                  PARSEC_PINS_FLAG method_flag,
                                  parsec_pins_callback cb_func,
                                  struct parsec_pins_next_callback_s* cb_data)
{
    if( method_flag >= PARSEC_PINS_FLAG_COUNT ) {
        parsec_warning("PINS register MUST be called with a valid event flag.");
        return PARSEC_ERR_BAD_PARAM;
    }
    if( NULL == cb_data ) {
        parsec_warning("PINS registration MUST be called with non-NULL data. Discard PINS module");
        return PARSEC_ERR_BAD_PARAM;
    }
    if (registration_disabled) {
        parsec_inform("PINS has been disabled by command line argument, causing this registration to have no effect.");
        return PARSEC_SUCCESS;
    }

    parsec_pins_next_callback_t* cb_event = &es->pins_events_cb[method_flag];

    *cb_data = *cb_event;

    cb_event->cb_func = cb_func;
    cb_event->cb_data = cb_data;

    return PARSEC_SUCCESS;
}

int parsec_pins_unregister_callback(struct parsec_execution_stream_s* es,
                                    PARSEC_PINS_FLAG method_flag,
                                    parsec_pins_callback cb,
                                    struct parsec_pins_next_callback_s** cb_data)
{
    *cb_data = NULL;
    if( method_flag >= PARSEC_PINS_FLAG_COUNT ) {
        parsec_warning("PINS unregister MUST be called with a valid event flag.");
        return PARSEC_ERR_BAD_PARAM;
    }
    if (registration_disabled) {
        parsec_inform("PINS has been disabled by command line argument, causing this UN-registration to have no effect.");
        return PARSEC_SUCCESS;
    }

    parsec_pins_next_callback_t* cb_event = &es->pins_events_cb[method_flag];
    while( (NULL != cb_event->cb_data) && (cb != cb_event->cb_func) ) {
        cb_event = cb_event->cb_data;
    }
    if( NULL == cb_event->cb_data ) {
        /* No matching event could be found in the list */
        return PARSEC_ERR_NOT_FOUND;
    }
    assert(cb_event->cb_func == cb);
    *cb_data = cb_event->cb_data;
    *cb_event = **cb_data;
    return PARSEC_SUCCESS;
}

PARSEC_PINS_FLAG parsec_pins_name_to_begin_flag(const char *name)
{
    PARSEC_PINS_FLAG val = PARSEC_PINS_FLAG_COUNT;
    if (0 == strncasecmp(name, "select", 6)) {
        val = SELECT_BEGIN;
    } else if (0 == strncasecmp(name, "prepare_input", 13)) {
        val = PREPARE_INPUT_BEGIN;
    } else if (0 == strncasecmp(name, "release_deps", 12)) {
        val = RELEASE_DEPS_BEGIN;
    } else if (0 == strncasecmp(name, "activate_cb", 11)) {
        val = ACTIVATE_CB_BEGIN;
    } else if (0 == strncasecmp(name, "data_flush", 10)) {
        val = DATA_FLUSH_BEGIN;
    } else if (0 == strncasecmp(name, "exec", 4)) {
        val = EXEC_BEGIN;
    } else if (0 == strncasecmp(name, "complete_exec", 13)) {
        val = COMPLETE_EXEC_BEGIN;
    } else if (0 == strncasecmp(name, "schedule", 8)) {
        val = SCHEDULE_BEGIN;
    }

    if (val == PARSEC_PINS_FLAG_COUNT) {
        parsec_warning("Unknown PINS task profiler event: %s\n", name);
    }

    return val;
}
