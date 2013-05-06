/* PaRSEC Performance Instrumentation Callback System */
#include <stdlib.h>
#include "pins.h"
#include "debug.h"

static int registration_disabled;

parsec_pins_callback * pins_array[PINS_FLAG_COUNT] = { 0 };

void parsec_instrument(PINS_FLAG method_flag, 
                 dague_execution_unit_t * exec_unit,
                 dague_execution_context_t * task, 
                 void * data) {
    (*(pins_array[method_flag]))(exec_unit, task, data);
}

/* convenience method provided 'just in case' */
void pins_disable_registration(int disable) {
	if (disable)
		DEBUG3(("PINS registration is disabled.\n"));
	else
		DEBUG3(("PINS registration is enabled.\n"));
	registration_disabled = disable;
}

/**
 The behavior of the PaRSEC PINS system is undefined if 
 pins_register_callback is not called at least once before 
 any call to parsec_instrument.
 */
parsec_pins_callback * pins_register_callback(PINS_FLAG method_flag, parsec_pins_callback * cb) {
    if (!pins_array[0]) {
        int i = 0;
        for (; i < PINS_FLAG_COUNT; i++) {
            if (pins_array[i] == NULL)
                pins_array[i] = &pins_empty_callback;
        }
        DEBUG(("Initialized PaRSEC PINS callbacks to empty_callback()"));
    }
    assert(cb != NULL);
    if (method_flag < PINS_FLAG_COUNT) {
	    if (registration_disabled) {
		    DEBUG2(("NOTE: PINS has been disabled by command line argument, causing this registration to fail."));
		    return NULL;
	    }
	    else {
		    parsec_pins_callback * prev = pins_array[method_flag];
		    pins_array[method_flag] = cb;
		    return prev;
	    }
    }
    else {
        DEBUG(("WARNING: Attempted to use the invalid flag %d with PaRSEC Performance Instrumentation!\n", method_flag));
    }
    return NULL;
}

parsec_pins_callback * pins_unregister_callback(PINS_FLAG method_flag) {
    if (method_flag < PINS_FLAG_COUNT) {
	    if (registration_disabled) {
		    DEBUG3(("NOTE: PINS has been disabled by command line argument, causing this UN-registration to fail."));
		    return NULL;
	    }
	    else {
		    parsec_pins_callback * prev = pins_array[method_flag];
		    pins_array[method_flag] = &pins_empty_callback;
		    return prev;
	    }
    }
    else {
        DEBUG(("WARNING: Attempted to use the invalid flag %d with PaRSEC Performance Instrumentation!\n", method_flag));
    }
    return NULL;
}

void pins_empty_callback(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data) {
    // do nothing
    (void) exec_unit;
    (void) task;
    (void) data;
}

