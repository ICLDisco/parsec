/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PINS_PAPI_UTILS_H
#define PINS_PAPI_UTILS_H

#include "dague.h"
#include "execution_unit.h"

#define NUM_DEFAULT_EVENTS 4  /* default number of events */

typedef struct parsec_pins_papi_values_s {
    long long values[NUM_DEFAULT_EVENTS];
} parsec_pins_papi_values_t;

typedef struct parsec_pins_papi_event_s {
    int        socket;
    int        core;
    int        pins_papi_native_event;
    int        frequency;
    char*      pins_papi_event_name;
} parsec_pins_papi_event_t;

typedef struct parsec_pins_papi_events_s {
    int                          num_counters;
    int                          num_allocated_counters;
    parsec_pins_papi_event_t*    events;
} parsec_pins_papi_events_t;

typedef struct parsec_pins_papi_callback_s {
    parsec_pins_next_callback_t  default_cb;
    int                          papi_eventset;
    int                          num_counters;
    int                          pins_prof_event[2];
    int                          begin_end;
    int                          num_tasks;
    int                          frequency;
    parsec_pins_papi_events_t*   events_list;
} parsec_pins_papi_callback_t;

/* CORES_PER_SOCKET is now in CMAKE config,
 * until dague-hwloc is updated to support dynamic determination */

int pins_papi_init(dague_context_t * master_context);
int pins_papi_fini(dague_context_t * master_context);
int pins_papi_thread_init(dague_execution_unit_t * exec_unit);
int pins_papi_thread_fini(dague_execution_unit_t * exec_unit);

/**
 * Parse a string into PAPI events and returns the event array initialized with the
 * set of translated PAPI event. Unknown or invalid events are ignored with no
 * further warning.
 */
parsec_pins_papi_events_t* parsec_pins_papi_events_new(char* events_list);

/**
 * Free a list of PAPI events.
 */
int parsec_pins_papi_events_free(parsec_pins_papi_events_t** pevents);

/**
 * Close all PAPI related structures from a cb event.
 */
void parsec_pins_papi_event_cleanup(parsec_pins_papi_callback_t* event_cb,
                                    parsec_pins_papi_values_t* pinfo);

#endif
