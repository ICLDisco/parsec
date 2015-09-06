/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include <papi.h>
#include "pins_papi_utils.h"
#include "dague/utils/output.h"
#include "dague/include/dague/os-spec-timing.h"

static int init_done = 0;
static int init_status = DAGUE_SUCCESS;
static const struct pins_papi_units_s {
    char const* unit_name[4];
    pins_papi_time_type_t unit_type;
    float unit_conversion_to_seconds;
} pins_papi_accepted_units[] = {
    { .unit_name = {"cycles", NULL},                     TIME_CYCLES, 0.0 },
    { .unit_name = {"nanosecond", "nano", "ns", NULL},   TIME_NS, 1e9 },
    { .unit_name = {"microsecond", "micro", "us", NULL}, TIME_US, 1e6 },
    { .unit_name = {"millisecond", "milli", "ms", NULL}, TIME_MS, 1e3 },
    { .unit_name = {"second", "sec", "s", NULL},         TIME_S, 1.0 } };

pins_papi_time_type_t system_units = 0;

static inline const struct pins_papi_units_s* find_unit_by_name(char* name)
{
    int i, j;

    for( i = 0; i < (sizeof(pins_papi_accepted_units) / sizeof(struct pins_papi_units_s)); j = 0, i++ ) {
        for( j = 0; NULL != pins_papi_accepted_units[i].unit_name[j]; j++ ) {
            if( 0 == strncmp(name, pins_papi_accepted_units[i].unit_name[j],
                             strlen(pins_papi_accepted_units[i].unit_name[j])) ) {
                return &pins_papi_accepted_units[i];
            }
        }
    }
    return NULL;
}

int find_unit_type_by_name(char* name, pins_papi_time_type_t* ptype)
{
    const struct pins_papi_units_s* unit = find_unit_by_name(name);
    if( NULL == unit ) {
        *ptype = TIME_CYCLES;
        return -1;
    }
    *ptype = unit->unit_type;
    return 0;
}

static inline const struct pins_papi_units_s* find_unit_by_type(pins_papi_time_type_t type)
{
    for( int i = 0; i < sizeof(pins_papi_accepted_units); i++ ) {
        if( pins_papi_accepted_units[i].unit_type == type ) {
            return &pins_papi_accepted_units[i];
        }
    }
    return NULL;
}

const char* find_unit_name_by_type(pins_papi_time_type_t type)
{
    const struct pins_papi_units_s* unit = find_unit_by_type(type);
    if( NULL == unit ) {
        return "unknown unit";
    }
    return unit->unit_name[0];
}


int convert_units(float *time, int source, int destination)
{
    const struct pins_papi_units_s *src, *dst;

    if( source == destination )  /* nothing to be done */
        return 0;
    src = find_unit_by_type(source);
    dst = find_unit_by_type(destination);
    if( (NULL == src) || (NULL == dst) )
        return -1;

    *time = *time * ((double)dst->unit_conversion_to_seconds /
                     (double)src->unit_conversion_to_seconds);
    return 0;
}

/**
 * This function should be called once per application in order to enable
 * a correct usage of PAPI.
 */
int pins_papi_init(dague_context_t * master_context)
{
    int i, err;

    (void)master_context;
    if( 0 == init_done++ ) {
        err = PAPI_library_init(PAPI_VER_CURRENT); /* this has to happen before threads get created */
        if( PAPI_VER_CURRENT != err ) {
            dague_output(0, "Failed to initialize PAPI (%x != %x[expected]). All components depending on PAPI will be disabled.",
                         PAPI_VER_CURRENT, err);
            init_status = -1;
            return -1;
        }
        dague_output(0, "Using PAPI version %x\n", PAPI_VER_CURRENT);
        /*PAPI_set_debug(PAPI_VERB_ECONT);*/
        err = PAPI_thread_init(( unsigned long ( * )( void ) ) ( pthread_self ));
        if( err != PAPI_OK ) {
            dague_output(0, "PAPI_thread_init failed (%s)! All components depending on PAPI will be disabled.\n", PAPI_strerror(err));
            init_status = -2;
            return -2;
        }
    }

    if( !find_unit_type_by_name(TIMER_UNIT, &system_units) ) {
        dague_output(0, "Could not find a propose time unit equivalent for %s. Fall back to %s\n",
                     TIMER_UNIT, find_unit_name_by_type(system_units));
    }

    return init_status;
}

/**
 * finalization function to be called once per application. It is the
 * counterpart fo pins_papi_init.
 */
int pins_papi_fini(dague_context_t * master_context)
{
    return DAGUE_SUCCESS;
}

/**
 * This function should be called by each thread in order to allow PAPI
 * to know about each of the potential users.
 */
int pins_papi_thread_init(dague_execution_unit_t * exec_unit)
{
    if( 0 == init_done ) return -1;

    (void)exec_unit;
    int err = PAPI_register_thread();
    if( err != PAPI_OK ) {
        dague_output(0, "PAPI_register_thread failed (%s). All components depending on PAPI will be disabled.\n", PAPI_strerror(err));
        return -2;
    }
    return 0;
}

/**
 * Function to be called once per thread, similar to pins_papi_thread_init.
 */

int pins_papi_thread_fini(dague_execution_unit_t * exec_unit)
{
    if( 0 == init_done ) return -1;

    (void)exec_unit;
    int err = PAPI_unregister_thread();
    if ( err != PAPI_OK )
        dague_output(0, "PAPI_unregister_thread failed (%s).\n", PAPI_strerror(err));
    return 0;
}

/**
 * Insert a new event into an already existing list of events. If the compact argument
 * is set, then the event will be matched with all the existing events in order to find
 * a similar class of event (core/uncore, same frequency and same PAPI component).
 * If the matching is succesful the new event will be chained to the previous events of
 * the same class. Otherwise, and this is also true when the compact argument is not
 * set, a new entry in the array will be create.
 *
 * Returns: 0 if the event has been succesfully inserted
 *         -1 in all other cases. The events have been left untouched.
 */
static int insert_event(parsec_pins_papi_events_t* events_array,
                        parsec_pins_papi_event_t* event,
                        int compact)
{
    void* tmp;
    int i;

    if( NULL != events_array->events ) {
        for( i = 0; i < events_array->num_counters; i++ ) {
            parsec_pins_papi_event_t* head = events_array->events[i];
            if( (NULL != head) &&
                (head->papi_location == event->papi_location) && (head->papi_component_index == event->papi_component_index) &&
                (head->papi_update_type == event->papi_update_type) ) {
                event->next = head;
                events_array->events[i] = event;
                return 0;
            }
        }
    }
    /* we failed to chain the event to a similar event family. Create a new one */
    if( events_array->num_counters == events_array->num_allocated_counters ) {
        events_array->num_allocated_counters <<= 1;  /* twice as big */
        tmp = realloc(events_array->events,  events_array->num_allocated_counters * sizeof(parsec_pins_papi_event_t*));
        if( NULL == tmp )  /* realloc failed */
            return -1;
        events_array->events = (parsec_pins_papi_event_t**)tmp;
        /* force the newly allocated events array to zero */
        for(i = events_array->num_counters; i < events_array->num_allocated_counters; i++) {
            events_array->events[i] = NULL;
        }
    }
    events_array->events[events_array->num_counters] = event;
    events_array->num_counters++;
    return 0;
}

parsec_pins_papi_events_t* parsec_pins_papi_events_new(char* events_str)
{
    char *mca_param_name, *token, *save_hptr = NULL;
    int err, tmp_eventset = PAPI_NULL;
    parsec_pins_papi_events_t* events = (parsec_pins_papi_events_t*)malloc(sizeof(parsec_pins_papi_events_t));
    parsec_pins_papi_event_t* event = NULL;
    PAPI_event_info_t papi_info;

    events->num_counters = 0;
    events->num_allocated_counters = 0;
    events->events = NULL;

    mca_param_name = strdup(events_str);
    token = strtok_r(mca_param_name, ",", &save_hptr);

    if( PAPI_OK != (err = PAPI_create_eventset(&tmp_eventset)) ) {
        dague_output(0, "%s: couldn't create the PAPI event set; ERROR: %s\n",
                     __func__, PAPI_strerror(err));
        return NULL;
    }

    while(token != NULL) {

        if( NULL == event ) {
            event = calloc( 1, sizeof(parsec_pins_papi_event_t) );
        } else {
            memset(event, 0, sizeof(parsec_pins_papi_event_t));
        }
        event->socket = -1;
        event->core = -1;
        event->frequency = 1;

        for(  /* none */; NULL != token;
                        token = strchr(token, (int)':'), token++ ) {

            if(token[0] == 'S') {
                if(token[1] != '*')
                    event->socket = atoi(&token[1]);
                continue;
            }
            if(token[0] == 'C') {
                if(token[1] != '*')
                    event->core = atoi(&token[1]);
                continue;
            }
            if(token[0] == 'F') {
                event->frequency_type = 0;  /* reset */
                event->frequency = 1;
                /* the remaining of this field must contain a number, which can be either
                 * a frequency or a time interval, and a unit. If the unit is missing then
                 * we have a frequency, otherwise we assume a timer.
                 */
                char* remaining;
                int value = (int)strtol(&token[1], &remaining, 10);
                if( remaining == &token[1] ) { /* no conversion was possible */
                    dague_output(0, "Impossible to convert the frequency [%s] of the PINS event %s. Assume frequency of 1.\n",
                                 &token[1], token);
                    continue;
                }
                if( value < 0 ) {
                    dague_output(0, "Obtained a negative value [%ld:%s] for the frequency of the PINS event %s. Assume frequency of 1.\n",
                                 value, &token[1], token);
                    continue;
                }
                const struct pins_papi_units_s* unit = find_unit_by_name(remaining);
                if( NULL != unit ) {
                    event->frequency_type = 1;
                    event->frequency = 1;
                    event->time = value;
                    convert_units(&event->time, unit->unit_type, system_units);
                }
                continue;
            }
            /* Make sure the event contains only valid values */
            if( event->frequency <= 0 ) {
                dague_output(0, "%s: Unsupported frequency (%d must be > 0). Discard the event.\n",
                             __func__, token);
                break;
            }

            /* Convert event name to code */
            if(PAPI_OK != (err = PAPI_event_name_to_code(token, &event->pins_papi_native_event)) ) {
                dague_output(0, "%s: Could not convert %s to a valid PAPI event name (%s). Ignore the event\n",
                             __func__, token, PAPI_strerror(err));
                break;
            }
            /* We're good to go, let's add the event to our queues */
            if( PAPI_OK != (err = PAPI_add_event(tmp_eventset,
                                                 event->pins_papi_native_event)) ) {
                /* Removing all events from an eventset does not reset the type of the eventset,
                 * generating errors when different classes of events are added. Thus, let's make
                 * sure we are not in this case.
                 */
                (void)PAPI_cleanup_eventset(tmp_eventset);  /* just do it and don't complain */
                if( PAPI_OK != (err = PAPI_add_event(tmp_eventset,
                                                     event->pins_papi_native_event)) ) {
                    dague_output(0, "%s: Unsupported event %s [%x](ERROR: %s). Discard the event.\n",
                                 __func__, token, event->pins_papi_native_event, PAPI_strerror(err));
                    break;
                }
            }
            if(event->frequency_type == 0){
                dague_output(0, "Valid PAPI event %s on socket %d (-1 for all), core %d (-1 for all) with frequency %d tasks\n",
                             token, event->socket, event->core, event->frequency);
            } else {
                dague_output(0, "Valid PAPI event %s on socket %d (-1 for all), core %d (-1 for all) with frequency %f %s\n",
                             token, event->socket, event->core, event->time, find_unit_name_by_type(system_units));
            }
            /* Remove the event to prevent issues with adding events from incompatible classes */
            PAPI_remove_event(tmp_eventset, event->pins_papi_native_event);

            if( PAPI_OK != (err = PAPI_get_event_info(event->pins_papi_native_event, &papi_info)) ) {
                dague_output(0, "%s: Impossible to extract information about event %s [%x](ERROR: %s). Discard the event.\n",
                             __func__, token, event->pins_papi_native_event, PAPI_strerror(err));
                break;
            }
            event->papi_component_index = papi_info.component_index;
            event->papi_location        = papi_info.location;
            event->papi_data_type       = papi_info.data_type;
            event->papi_update_type     = papi_info.update_type;

            /* We have a valid event, let's move to the next */
            event->pins_papi_event_name = strdup(token);
            /* We now have a valid event ready to be monitored */
            insert_event(events, event, true);
            event = NULL;
            break;  /* the internal loop should not be completed */
        }
        token = strtok_r(NULL, ",", &save_hptr);
    }

    free(mca_param_name);
    if( PAPI_NULL != tmp_eventset ) {
        (void)PAPI_cleanup_eventset(tmp_eventset);  /* just do it and don't complain */
        (void)PAPI_destroy_eventset(&tmp_eventset);
    }
    return events;
}

int parsec_pins_papi_events_free(parsec_pins_papi_events_t** pevents)
{
    parsec_pins_papi_events_t* events = *pevents;
    parsec_pins_papi_event_t* event;
    int i;

    for( i = 0; i < events->num_counters; i++ ) {
        event = events->events[i];
        if( NULL != event->pins_papi_event_name )
            free(event->pins_papi_event_name);
    }
    free(events->events); events->events = NULL;
    free(events);
    *pevents = NULL;
    return 0;
}

void parsec_pins_papi_event_cleanup(parsec_pins_papi_callback_t* event_cb,
                                    parsec_pins_papi_values_t* pinfo)
{
    int err;

    if(PAPI_NULL != event_cb->papi_eventset) {
        if( PAPI_OK != (err = PAPI_stop(event_cb->papi_eventset, pinfo->values)) ) {
            dague_output(0, "couldn't stop PAPI eventset ERROR: %s\n",
                         PAPI_strerror(err));
        }
        if( PAPI_OK != (err = PAPI_cleanup_eventset(event_cb->papi_eventset)) ) {
            dague_output(0, "failed to cleanup eventset (ERROR: %s)\n", PAPI_strerror(err));
        }

        if( PAPI_OK != (err = PAPI_destroy_eventset(&event_cb->papi_eventset)) ) {
            dague_output(0, "failed to destroy PAPI eventset (ERROR: %s)\n", PAPI_strerror(err));
        }
        event_cb->papi_eventset = PAPI_NULL;
    }
}
