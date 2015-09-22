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
#include "dague/include/dague/os-spec-timing.h"

static char* mca_param_string;
static parsec_pins_papi_events_t* pins_papi_events = NULL;

extern pins_papi_time_type_t system_units;

/**
 * Determines which event groups within the events list in event_cb need to be read, and
 * supplies the appropriate portion of the array from PAPI_read to the dague_profiling_trace
 * function, which pushes this information into the global profiling information.
 */
static inline int
pins_papi_read_and_trace(dague_execution_unit_t* exec_unit,
                         parsec_pins_papi_callback_t* event_cb)
{
    parsec_pins_papi_values_t info;
    int err, i;

    /* Read the current values of the counters in event_cb's eventset and store them in info.values */
    if( PAPI_OK != (err = PAPI_read(event_cb->papi_eventset, info.values)) ) {
        dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        return DAGUE_ERROR;
    }

    int index = 0; /* The starting index of a group in the events list */
    uint64_t check = 1; /* The bit to check against for whether a group is measured (starting from the right-most bit) */
    for(i = 0; i < event_cb->num_groups; i++) {
        /* We need to read this group */
        if((event_cb->to_read & check) != 0) { /* This AND operation determines whether the bit corresponding to 'check' is set to 1 */
            (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->groups[i].pins_prof_event[event_cb->groups[i].begin_end],
                                        45, 0, (void*)&info.values[index]);
            event_cb->groups[i].begin_end = (event_cb->groups[i].begin_end + 1) & 0x1;  /* aka. % 2 */
        }
        index += event_cb->groups[i].num_counters; /* Move the index to the next group. */
        check <<= 1; /* Move the set bit to the left one to check the next group */
    }
    event_cb->to_read = 0; /* Reset the bit string to all 0's */
    return DAGUE_SUCCESS;
}

/**
 * This function is called many times throughout the program's execution.  It is typically called
 * before and/or after each task (after each task is the standard pattern).  This function determines
 * whether or not a PAPI_read is required for any of the frequency groups in the event_cb's events list,
 * and if so calls pins_papi_read_and_trace to read them and put the information in the profiling data.
 */
static void pins_papi_trace(dague_execution_unit_t* exec_unit,
                           dague_execution_context_t* exec_context,
                           parsec_pins_next_callback_t* cb_data)
{
    parsec_pins_papi_callback_t* event_cb = (parsec_pins_papi_callback_t*)cb_data;
    dague_time_t current_time = take_time();
    bool read = false;
    int i;

    uint64_t set = 1; /* The bit to set to 1 (starting from the right-most bit) */
    for(i = 0; i < event_cb->num_groups; i++) {
        if(event_cb->groups[i].frequency < 0) { /* time-based frequency */
            float elapsed_time = (float)diff_time(event_cb->groups[i].start_time, current_time);

            /* The elapsed time is longer than the frequency period, so we read. */
            if(elapsed_time > event_cb->groups[i].time){
                DAGUE_OUTPUT((0, "[Thread %d] Elapsed Time: %f (%s) > %f\n", exec_unit->th_id, elapsed_time,
                              find_unit_name_by_type(system_units), event_cb->groups[i].time));
                event_cb->groups[i].start_time = current_time;

                read = true; /* We need a call to pins_papi_read_and_trace() */
                event_cb->to_read |= set; /* This sets the appropriate bit in to_read to 1 */
            }
        } else { /* task-based frequency */
            /* This group had only one task left in its countdown, so we read and reset the countdown. */
            if(1 == event_cb->groups[i].trigger ) {
                read = true; /* We need a call to pins_papi_read_and_trace() */
                event_cb->to_read |= set; /* This sets the appropriate bit in to_read to 1 */
                event_cb->groups[i].trigger = event_cb->groups[i].frequency;
            } else event_cb->groups[i].trigger--;
        }
        set <<= 1; /* Move the set bit to the left one to check the next group */
    }

    /* Actually call the read function to read all of the relevant groups. */
    if(read) {
        (void)pins_papi_read_and_trace(exec_unit, event_cb);
    }
}

static void pins_init_papi(dague_context_t * master_context)
{
    pins_papi_init(master_context);

    dague_mca_param_reg_string_name("pins", "papi_event",
                                    "PAPI events to be gathered at both socket and core level with task and/or time based frequency.\n",
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

/**
 * This function creates the dictionary entries for each frequency group within the events list,
 * starts the eventset for the event_cb, and registers the EXEC_BEGIN (if needed) and EXEC_END callbacks.
 */
static int register_event_cb(dague_execution_unit_t * exec_unit,
                             parsec_pins_papi_callback_t* event_cb,
                             char** conv_string,
                             int event_id)
{
    char* key_string;
    int err, i;

    assert( NULL != event_cb );

    event_cb->to_read = 0; /* Sets the bit string denoting whether groups should be read to all 0's */
    /* Create the dictionary entries for each frequency group. */
    for(i = 0; i < event_cb->num_groups; i++)
    {
        if(event_cb->groups[i].frequency > 0) { /* task-based frequency */
            asprintf(&key_string, "PINS_PAPI_S%d_C%d_F%dt_%d",
                     exec_unit->socket_id, exec_unit->core_id, event_cb->groups[i].frequency, event_id);
        }
        else { /* time-based frequency */
            asprintf(&key_string, "PINS_PAPI_S%d_C%d_F%f%s_%d",
                     exec_unit->socket_id, exec_unit->core_id, event_cb->groups[i].time, find_short_unit_name_by_type(system_units), event_id);
        }
        dague_profiling_add_dictionary_keyword(key_string, "fill:#00AAFF",
                                               sizeof(long long) * event_cb->groups[i].num_counters,
                                               conv_string[i],
                                               &event_cb->groups[i].pins_prof_event[0],
                                               &event_cb->groups[i].pins_prof_event[1]);
        free(key_string);
    }

    /* Start the PAPI eventset */
    if( PAPI_OK != (err = PAPI_start(event_cb->papi_eventset)) ) {
        dague_output(0, "couldn't start PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        event_cb->num_counters = 0;
        return DAGUE_ERROR;
    }

    /* Set all of the frequency groups' start times to now */
    dague_time_t start_time = take_time();
    for(i = 0; i < event_cb->num_groups; i++) {
        event_cb->groups[i].start_time = start_time;
    }

    /* the event is now ready. Trigger it once ! */
    if( DAGUE_SUCCESS != (err = pins_papi_read_and_trace(exec_unit, event_cb)) ) {
        dague_output(0, "PAPI event %s core %d socket %d frequency %d failed to generate. Disabled!\n",
                     conv_string[i], event_cb->event->core, event_cb->event->socket, event_cb->event->frequency);
        return err;
    }

    bool need_begin = false;
    for(i = 0; i < event_cb->num_groups; i++) {
        if(event_cb->groups[i].frequency > 0){ /* task-based frequency */
            dague_output(0, "PAPI event %s core %d socket %d frequency %d tasks enabled\n",
                         conv_string[i], event_cb->event->core, event_cb->event->socket, event_cb->groups[i].frequency);
            if(event_cb->groups[i].frequency == 1) /* An EXEC_BEGIN needs to be registered. */
                need_begin = true;
        } else { /* time-based frequency */
            dague_output(0, "PAPI event %s core %d socket %d frequency %f %s enabled\n",
                         conv_string[i], exec_unit->core_id, exec_unit->socket_id, event_cb->groups[i].time,
                         find_short_unit_name_by_type(system_units));
        }
    }

    /* Register the EXEC_BEGIN and EXEC_END callbacks */
    if(need_begin) {
        PINS_REGISTER(exec_unit, EXEC_BEGIN, pins_papi_trace,
                      (parsec_pins_next_callback_t*)event_cb);
    }
    PINS_REGISTER(exec_unit, EXEC_END, pins_papi_trace,
                  (parsec_pins_next_callback_t*)event_cb);

    return DAGUE_SUCCESS;
}

/**
 * This function creates the event_cb object for this thread and performs other
 * thread-specific initializations.  This function also creates strings for each
 * group with the event names and data types for use in registering the event_cb's
 */
static void pins_thread_init_papi(dague_execution_unit_t * exec_unit)
{
    parsec_pins_papi_callback_t* event_cb = NULL;
    parsec_pins_papi_event_t* event;
    parsec_pins_papi_values_t info;
    int i, my_socket, my_core, err, event_id = 0;
    char **conv_string = NULL, *datatype;

    if( NULL == pins_papi_events ) /* There aren't any events, so nothing to do. */
        return;

    my_socket = exec_unit->socket_id;
    my_core = exec_unit->core_id;

    pins_papi_thread_init(exec_unit);

    /* Iterate through all of the event classes */
    for( i = 0; i < pins_papi_events->num_counters; i++ ) {
        event = pins_papi_events->events[i];
        conv_string = NULL;

        /* Iterate through all of the events in the event class. */
        for( ; NULL != event; event = event->next ) {
            /* Is this event supposed to be measured on this socket? */
            if( (event->socket != -1) && (event->socket != my_socket) )
                continue;
            /* Is this event supposed to be measured on this core? */
            if( (event->core != -1) && (event->core != my_core) )
                continue;

            if( NULL == event_cb ) {  /* create the event and the PAPI eventset */
                event_cb = (parsec_pins_papi_callback_t*)malloc(sizeof(parsec_pins_papi_callback_t));
                event_cb->groups = (parsec_pins_papi_frequency_group_t*)malloc(sizeof(parsec_pins_papi_frequency_group_t));

                event_cb->papi_eventset          = PAPI_NULL;
                event_cb->num_counters           = 0;
                event_cb->num_groups             = 0;
                event_cb->event                  = event;
                event_cb->groups[0].group_num    = event->group;
                event_cb->groups[0].num_counters = 0;
                event_cb->groups[0].frequency    = event->frequency;
                event_cb->groups[0].trigger      = event->frequency;
                event_cb->groups[0].time         = event->time;
                event_cb->groups[0].begin_end    = 0;

                /* Create an empty eventset */
                if( PAPI_OK != (err = PAPI_create_eventset(&event_cb->papi_eventset)) ) {
                    dague_output(0, "%s: thread %d couldn't create the PAPI event set; ERROR: %s\n",
                                 __func__, exec_unit->th_id, PAPI_strerror(err));
                    /* Destroy the event it is unsafe to use */
                    free(event_cb->groups);
                    free(event_cb); event_cb = NULL;
                    continue;
                }
            }

            /* Add the event to the eventset */
            if( PAPI_OK != (err = PAPI_add_event(event_cb->papi_eventset,
                                                 event->pins_papi_native_event)) ) {
                dague_output(0, "%s: failed to add event %s; ERROR: %s\n",
                             __func__, event->pins_papi_event_name, PAPI_strerror(err));
                if(event_cb->num_groups == 0){
                    free(event_cb->groups);
                    event_cb->groups = NULL;
                }
                continue;
            }

            /* At this point, we've added an event */
            event_cb->num_counters++;
            if(event_cb->num_groups == 0){
                event_cb->num_groups = 1;
            }

            bool new_group = false;
            /* We need to create a new group. */
            if(event_cb->groups[event_cb->num_groups-1].group_num != event->group) {
                int group_index = event_cb->num_groups;
                event_cb->num_groups++;
                void *tmp;
                tmp = realloc(event_cb->groups, event_cb->num_groups * sizeof(parsec_pins_papi_frequency_group_t));
                if(tmp == NULL){
                    dague_output(0, "Failed to add a new frequency group for event (%s).\n",
                                 event->pins_papi_event_name);
                    return;
                }
                event_cb->groups = (parsec_pins_papi_frequency_group_t*)tmp;

                event_cb->groups[group_index].num_counters = 1;
                event_cb->groups[group_index].group_num = event->group;
                event_cb->groups[group_index].begin_end = 0;
                event_cb->groups[group_index].frequency = event->frequency;
                event_cb->groups[group_index].trigger = event->frequency;
                event_cb->groups[group_index].time = event->time;
                new_group = true;
            }
            else { /* Increment the current group. */
                event_cb->groups[event_cb->num_groups-1].num_counters++;
            }

            switch( event->papi_data_type ) {
            case PAPI_DATATYPE_INT64: datatype = "int64_t"; break;
            case PAPI_DATATYPE_UINT64: datatype = "uint64_t"; break;
            case PAPI_DATATYPE_FP64: datatype = "double"; break;
            case PAPI_DATATYPE_BIT64: datatype = "int64_t"; break;
            default: datatype = "int64_t"; break;
            }

            /* Update the conv_string */
            if( NULL == conv_string ) {
                conv_string = (char**)malloc(sizeof(char*));
                asprintf(&conv_string[0], "%s{%s}"PARSEC_PINS_SEPARATOR, event->pins_papi_event_name, datatype);
            }
            else {
                if(new_group){
                    void *temp_string;
                    temp_string = realloc(conv_string, event_cb->num_groups * sizeof(char*));
                    if(temp_string == NULL) {
                        dague_output(0, "Failed to resize the events string\n");
                        return;
                    }
                    conv_string = (char**)temp_string;
                    asprintf(&conv_string[event_cb->num_groups-1], "%s{%s}"PARSEC_PINS_SEPARATOR, event->pins_papi_event_name, datatype);
                }
                else{
                    char* tmp = conv_string[event_cb->num_groups-1];
                    asprintf(&conv_string[event_cb->num_groups-1], "%s%s{%s}"PARSEC_PINS_SEPARATOR, tmp, event->pins_papi_event_name, datatype);
                    free(tmp);
                }
            }
        }
    }

    /* Shouldn't 'register_event_cb()' happen within the outer for loop before conv_string becomes NULL again? */
    if( NULL != event_cb ) {
        if( DAGUE_SUCCESS != (err = register_event_cb(exec_unit, event_cb, conv_string, event_id)) ) {
            parsec_pins_papi_event_cleanup(event_cb, &info);
            free(event_cb->groups);
            while(event_cb->event != NULL) {
                parsec_pins_papi_event_t* temp_event = event_cb->event->next;
                free(event_cb->event);
                event_cb->event = temp_event;
            }
            free(event_cb);
        }
    }
    if( NULL != conv_string ) {
        for(i = 0; i < event_cb->num_groups; i++)
            free(conv_string[i]);
        free(conv_string);
    }
}

/**
 * This function breaks down the event_cb and frees up thread-specific data.  This funciton also
 * performs the final traces for each frequency group within the event_cb.
 */
static void pins_thread_fini_papi(dague_execution_unit_t* exec_unit)
{
    parsec_pins_papi_callback_t* event_cb;
    parsec_pins_papi_values_t info;
    int i;

    do {
        /* Should this be in a loop to unregister all the callbacks for this thread? */
        PINS_UNREGISTER(exec_unit, EXEC_END, pins_papi_trace, (parsec_pins_next_callback_t**)&event_cb);
        if( NULL == event_cb )
            return;

        bool has_begin = false;
        /* Determine whether EXEC_BEGIN was registered */
        for(i = 0; i < event_cb->num_groups, i++) {
            if(event_cb->groups[i].frequency == 1) {
                has_begin = true;
                break;
            }
        }
        /* EXEC_BEGIN was registered, so unregister it. */
        if( has_begin ) {  /* this must have an EXEC_BEGIN */
            parsec_pins_papi_callback_t* start_cb;
            PINS_UNREGISTER(exec_unit, EXEC_BEGIN, pins_papi_trace, (parsec_pins_next_callback_t**)&start_cb);
            if( NULL == start_cb ) {
                dague_output(0, "Unsetting exception of an event with frequency 1 but without a start callback. Ignored.\n");
            }
        }
        /* Clean up the eventset, and perform final recordings. */
        if( PAPI_NULL != event_cb->papi_eventset ) {
            parsec_pins_papi_event_cleanup(event_cb, &info);

            int index = 0;
            for(i = 0; i < event_cb->num_groups; i++) {
                /* If the last profiling event was an 'end' event */
                if(event_cb->groups[i].begin_end == 0) {
                    (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->groups[i].pins_prof_event[0],
                                                45, 0, (void *)&info.values[index]);
                }
                (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->groups[i].pins_prof_event[1],
                                            45, 0, (void *)&info.values[index]);
                index += event_cb->groups[i].num_counters;
            }
        }
        free(event_cb);
    } while(1);

    pins_papi_thread_fini(exec_unit);
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
