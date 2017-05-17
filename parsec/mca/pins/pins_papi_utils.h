/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PINS_PAPI_UTILS_H
#define PINS_PAPI_UTILS_H

#include "parsec.h"
#include "parsec/execution_unit.h"
#include "parsec/include/parsec/os-spec-timing.h"

typedef struct parsec_pins_papi_values_s {
    long long values[1];
} parsec_pins_papi_values_t;

typedef enum {TIME_CYCLES, TIME_NS,TIME_US, TIME_MS, TIME_S} pins_papi_time_type_t;
extern pins_papi_time_type_t system_units;

typedef struct parsec_pins_papi_event_s {
    int                              socket;
    int                              core;
    int                              pins_papi_native_event;
    int                              group;
    int                              frequency;
    float                            time;
    char*                            pins_papi_event_name;
    int                              papi_component_index;
    int                              papi_location;
    int                              papi_data_type;
    int                              papi_update_type;
    struct parsec_pins_papi_event_s* next;
} parsec_pins_papi_event_t;

typedef struct parsec_pins_papi_frequency_group_s {
    int          group_num;
    int          num_counters;
    int          pins_prof_event[2];
    int          begin_end;
    int          frequency;
    int          trigger;
    float        time;
    parsec_time_t start_time;
} parsec_pins_papi_frequency_group_t;

typedef struct parsec_pins_papi_events_s {
    int                        num_counters;
    int                        num_allocated_counters;
    parsec_pins_papi_event_t** events;
} parsec_pins_papi_events_t;

typedef struct parsec_pins_papi_callback_s {
    parsec_pins_next_callback_t         default_cb;
    int                                 papi_eventset;
    int                                 num_counters;
    int                                 num_groups;
    uint64_t                            to_read;
    parsec_pins_papi_frequency_group_t* groups;
    parsec_pins_papi_event_t*           event;
} parsec_pins_papi_callback_t;

int pins_papi_init(parsec_context_t * master_context);
int pins_papi_fini(parsec_context_t * master_context);
int pins_papi_thread_init(parsec_execution_unit_t * exec_unit);
int pins_papi_thread_fini(parsec_execution_unit_t * exec_unit);

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

/* Functions to manipulate timing units */
extern const char* find_unit_name_by_type(pins_papi_time_type_t type);
extern int find_unit_type_by_name(char* name, pins_papi_time_type_t* ptype);
extern int convert_units(float *time, pins_papi_time_type_t source, pins_papi_time_type_t destination);
extern const char* find_short_unit_name_by_type(pins_papi_time_type_t type);

#endif
