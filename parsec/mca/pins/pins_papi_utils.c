/*
 * Copyright (c) 2012-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "parsec/parsec_config.h"
#include <papi.h>
#include "pins_papi_utils.h"
#include "parsec/utils/output.h"
#include "parsec/include/parsec/os-spec-timing.h"
#include "parsec/utils/debug.h"

static int init_done = 0;
static int init_status = 0;

/**
 * A structure to store the unit conversion information for the
 * time-based frequency events.
 */
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

pins_papi_time_type_t system_units = TIME_CYCLES;

/**
 * A utility function for finding the pins_papi_units_s structure in the
 * pins_papi_accepted_units array that corresponds to 'name'.
 */
static inline const struct pins_papi_units_s* find_unit_by_name(char* name)
{
    uint32_t i, j;

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

/**
 * A utility function for finding the the correct pins_papi_time_type_t
 * enumeration for 'name' and storing it in 'ptype'
 */
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

/**
 * A utility function for finding the pins_papi_units_s structure in the
 * pins_papi_accepted_units array that corresponds to 'type'.
 */
static inline const struct pins_papi_units_s* find_unit_by_type(pins_papi_time_type_t type)
{
    for( uint32_t i = 0; i < sizeof(pins_papi_accepted_units); i++ ) {
        if( pins_papi_accepted_units[i].unit_type == type ) {
            return &pins_papi_accepted_units[i];
        }
    }
    return NULL;
}

/**
 * A utility function for finding the full name of the unit type
 * denoted by 'type'.
 */
const char* find_unit_name_by_type(pins_papi_time_type_t type)
{
    const struct pins_papi_units_s* unit = find_unit_by_type(type);
    if( NULL == unit ) {
        return "unknown unit";
    }
    return unit->unit_name[0];
}

/**
 * A utility function for finding the short name of the unit type
 * denoted by 'type'.
 */
const char* find_short_unit_name_by_type(pins_papi_time_type_t type)
{
    const struct pins_papi_units_s* unit = find_unit_by_type(type);
    if( NULL == unit ) {
        return "unknown unit";
    }
    if( unit->unit_type == TIME_CYCLES )
        return unit->unit_name[0];
    return unit->unit_name[2];
}

/**
 * Converts time from the units denoted by 'source' to the units denoted
 * by 'destination' and stores the result in 'time'.  Returns -1 on failure
 * and 0 on success.
 */
int convert_units(float *time, pins_papi_time_type_t source, pins_papi_time_type_t destination)
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
int pins_papi_init(parsec_context_t * master_context)
{
    int err;

    if( 0 == init_done++ ) {
        err = PAPI_library_init(PAPI_VER_CURRENT); /* this has to happen before threads get created */
        if( PAPI_VER_CURRENT != err ) {
            parsec_warning("Failed to initialize PAPI (%x != %x[expected]). All components depending on PAPI will be disabled.",
                         PAPI_VER_CURRENT, err);
            init_status = -1;
            return -1;
        }
        parsec_debug_verbose(4, parsec_debug_output, "Using PAPI version %x", PAPI_VER_CURRENT);
        /*PAPI_set_debug(PAPI_VERB_ECONT);*/
        err = PAPI_thread_init(( unsigned long ( * )( void ) ) ( pthread_self ));
        if( err != PAPI_OK ) {
            parsec_warning("PAPI_thread_init failed (%s)! All components depending on PAPI will be disabled.", PAPI_strerror(err));
            init_status = -2;
            return -2;
        }
    }

    /* If the TIMER_UNIT variable has been set, we use those units
     * otherwise we stick with the default system units.
     */
    if( !find_unit_type_by_name(TIMER_UNIT, &system_units) ) {
        parsec_debug_verbose(4, parsec_debug_output, "Could not find a proposed time unit equivalent for %s. Fall back to %s",
                     TIMER_UNIT, find_unit_name_by_type(system_units));
    }

    (void)master_context;
    return init_status;
}

/**
 * finalization function to be called once per application. It is the
 * counterpart fo pins_papi_init.
 */
int pins_papi_fini(parsec_context_t * master_context)
{
    (void)master_context;
    return 0;
}

/**
 * This function should be called by each thread in order to allow PAPI
 * to know about each of the potential users.
 */
int pins_papi_thread_init(parsec_execution_stream_t* es)
{
    if( 0 == init_done ) return -1;

    (void)es;
    int err = PAPI_register_thread();
    if( err != PAPI_OK ) {
        parsec_warning("PAPI_register_thread failed (%s). All components depending on PAPI will be disabled.", PAPI_strerror(err));
        return -2;
    }
    return 0;
}

/**
 * Function to be called once per thread, similar to pins_papi_thread_init.
 */

int pins_papi_thread_fini(parsec_execution_stream_t* es)
{
    if( 0 == init_done ) return -1;

    (void)es;
    int err = PAPI_unregister_thread();
    if ( err != PAPI_OK )
        parsec_debug_verbose(3, parsec_debug_output, "PAPI_unregister_thread failed (%s).\n", PAPI_strerror(err));
    return 0;
}

/**
 * Insert a new event into an already existing list of events. If the compact argument
 * is set, then the event will be matched with all the existing events in order to find
 * a similar class of event (same PAPI component) and a frequency group within that class.
 * If the matching is succesful the new event will be chained to the previous events of
 * the same class. Otherwise, and this is also true when the compact argument is not
 * set, a new entry in the array will be created.
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

    /* If there are existing event classes... */
    if( NULL != events_array->events ) {
        /* Iterate through all of the event classes to find a PAPI-compatible class. */
        for( i = 0; i < events_array->num_counters; i++ ) {
            parsec_pins_papi_event_t* head = events_array->events[i];
            /* The events are PAPI compatible */
            if( (NULL != head) &&
                (head->papi_location == event->papi_location) && (head->papi_component_index == event->papi_component_index) &&
                (head->papi_update_type == event->papi_update_type) ) {
                int group;
                /* Iterate through all of the groups in this event class to find a
                 * frequency-compatible group, or add a new group.
                 */
                while(head != NULL){
                    group = head->group;
                    /* The events are frequency compatible (task-based and same frequency) */
                    if( (event->frequency > 0 && head->frequency > 0) &&
                        event->frequency == head->frequency ) {
                        event->group = group;
                        event->next = head->next;
                        head->next = event;
                        return 0;
                    }
                    /* The events are frequency compatible (time-based and same time) */
                    if( (event->frequency < 0 && head->frequency < 0) &&
                        event->time == head->time ) {
                        event->group = group;
                        event->next = head->next;
                        head->next = event;
                        return 0;
                    }
                    /* The event isn't frequency compatible with any current events, create a new frequency group. */
                    if(head->next == NULL){
                        event->group = group + 1;
                        head->next = event;
                        return 0;
                    }
                    head = head->next;
                }
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
    event->group = 0;
    events_array->events[events_array->num_counters] = event;
    events_array->num_counters++;
    (void)compact;
    return 0;
}

/**
 * Parses the 'events_str' string from the user-specified mca parameter settings,
 * and adds these events to an events list for this thread if they are valid PAPI
 * events and have a valid socket, core, and frequency specified.
 *
 * The parser accepts lines like:
 *    <Specifiers><CounterName0>,<Specifiers><CouterName1>...
 * Where <CounterName*> are counter names, and can contain
 * any character except ',' and <Specifiers> is
 *    [C<number>:][S<number>:][F<frequency>:] in any order.
 *  <number> or <frequency> can be * to denote any
 *  <number> is otherwise a number
 *  <frequency> is a positive floating point number followed by a time unit
 *  - Any specifier can be set only once per counter
 *  - If the counter name starts with S, C or F, the corresponding specifier
 *    must be set before providing the counter name
 *  - If the counter name contains a ':', the three specifiers must be
 *    set before providing the counter name
 *
 * Returns: A valid events list on success
 *          NULL in all other cases.
 */
parsec_pins_papi_events_t* parsec_pins_papi_events_new(char* events_str)
{
    char *mca_param_name, *token, *save_hptr = NULL;
    int err, tmp_eventset = PAPI_NULL;
    parsec_pins_papi_events_t* events = (parsec_pins_papi_events_t*)malloc(sizeof(parsec_pins_papi_events_t));
    parsec_pins_papi_event_t* event = NULL;
    PAPI_event_info_t papi_info;
    int socket_set, core_set, frequency_set;

    events->num_counters = 0;
    events->num_allocated_counters = 1;
    /* Create a placeholder event */
    events->events = (parsec_pins_papi_event_t**)malloc(sizeof(parsec_pins_papi_event_t*));

    mca_param_name = strdup(events_str);
    token = strtok_r(mca_param_name, ",", &save_hptr);

    /* Create a temporary eventset for checking whether events are valid. */
    if( PAPI_OK != (err = PAPI_create_eventset(&tmp_eventset)) ) {
        parsec_warning( "Couldn't create the PAPI event set %s",
                       PAPI_strerror(err));
        return NULL;
    }

    /* Iterate through the mca_param_name string to identify events and test them. */
    while(token != NULL) {
        /* Reset the memory in the event so we're starting fresh. */
        if( NULL == event ) {
            event = calloc( 1, sizeof(parsec_pins_papi_event_t) );
        } else {
            memset(event, 0, sizeof(parsec_pins_papi_event_t));
        }
        socket_set = 0;
        core_set = 0;
        frequency_set = 0;

        event->socket = -1;
        event->core = -1;
        event->frequency = 1;

        /* Iterate through the separate events in the string that are separated by a '|' character. */
        for(  /* none */; NULL != token;
                        token = strchr(token, (int)'|'), token++ ) {
            /* This token represents the socket for this event. */
            if((0 == socket_set) && token[0] == 'S') {
                if(token[1] != '*') {
                    errno = 0;
                    event->socket = strtol(&token[1], NULL, 10);
                    if( 0 != errno) {  /* failed to convert. Assume we are looking at an event. */
                        parsec_debug_verbose(10, parsec_debug_output, "Impossible to convert the socket [%s] of the PINS event %s. "
                                             "Assume this is the name of the event", &token[1], token);
                        goto find_event;
                    }
                }
                socket_set = 1;
                continue;
            }
            /* This token represents the core for this event. */
            if((0 == core_set) && token[0] == 'C') {
                if(token[1] != '*') {
                    errno = 0;
                    event->core = strtol(&token[1], NULL, 10);
                    if( 0 != errno) {  /* failed to convert. Assume we are looking at an event. */
                        parsec_debug_verbose(10, parsec_debug_output, "Impossible to convert the core [%s] of the PINS event %s. "
                                             "Assume this is the name of the event", &token[1], token);
                        goto find_event;
                    }
                }
                core_set = 1;
                continue;
            }

            /* This token represents the frequency for this event, so we need to determine
             * whether this is a task-based or time-based frequency.  If it is a time-based
             * frequency, we must determine the units and convert the units specified into
             * the units used by this system.
             */
            if((0 == frequency_set) && token[0] == 'F') {
                event->frequency = 1;
                frequency_set = 1;
                /* the remaining of this field must contain a number, which can be either
                 * a frequency or a time interval, and a unit. If the unit is missing then
                 * we have a frequency, otherwise we assume a timer.
                 */
                char* remaining;
                float value = strtof(&token[1], &remaining);
                if( remaining == &token[1] ) { /* no conversion was possible */
                    parsec_debug_verbose(3, parsec_debug_output, "Impossible to convert the frequency [%s] of the PINS event %s. "
                                         "Assume this is the name of the event", &token[1], token);
                    goto find_event;
                }
                if( value < 0 ) {
                    parsec_debug_verbose(3, parsec_debug_output, "Obtained a negative value [%ld:%s] for the frequency of the PINS event %s. Assume frequency of 1.",
                                 value, &token[2], token);
                    continue;
                }
                const struct pins_papi_units_s* unit = find_unit_by_name(remaining);
                if( NULL != unit ) {
                    event->frequency = -1;
                    event->time = value;
                    convert_units(&event->time, unit->unit_type, system_units);
                }
                else {
                    event->frequency = (int)value;
                    parsec_debug_verbose(3, parsec_debug_output, "No units found.  Assuming task-based frequency: %d", event->frequency);
                }
                continue;
            }

        find_event:
            /* Convert event name to code */
            if(PAPI_OK != (err = PAPI_event_name_to_code(token, &event->pins_papi_native_event)) ) {
                parsec_warning("Could not convert %s to a valid PAPI event name (%s). Ignore the event",
                               token, PAPI_strerror(err));
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
                    parsec_debug_verbose(3, parsec_debug_output, "Unsupported event %s [%x](ERROR: %s). Discard the event.",
                                 token, event->pins_papi_native_event, PAPI_strerror(err));
                    break;
                }
            }
            if(event->frequency > 0){
                parsec_debug_verbose(10, parsec_debug_output, "Valid PAPI event %s on socket %d (-1 for all), core %d (-1 for all) with frequency %d tasks",
                             token, event->socket, event->core, event->frequency);
            } else {
                parsec_debug_verbose(10, parsec_debug_output, "Valid PAPI event %s on socket %d (-1 for all), core %d (-1 for all) with frequency %f %s",
                             token, event->socket, event->core, event->time, find_unit_name_by_type(system_units));
            }
            /* Remove the event to prevent issues with adding events from incompatible classes */
            PAPI_remove_event(tmp_eventset, event->pins_papi_native_event);

            if( PAPI_OK != (err = PAPI_get_event_info(event->pins_papi_native_event, &papi_info)) ) {
                parsec_debug_verbose(3, parsec_debug_output, "Impossible to extract information about event %s [%x] (error %s). Discard the event.",
                              token, event->pins_papi_native_event, PAPI_strerror(err));
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

    /* If we didn't add any events, there's no need to keep the placeholder event around */
    if(events->num_counters == 0){
        free(events->events);
        events->events = NULL;
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
            parsec_debug_verbose(3, parsec_debug_output, "Couldn't stop PAPI eventset (error %s)",
                         PAPI_strerror(err));
        }
        if( PAPI_OK != (err = PAPI_cleanup_eventset(event_cb->papi_eventset)) ) {
            parsec_debug_verbose(3, parsec_debug_output, "Failed to cleanup eventset (error %s)", PAPI_strerror(err));
        }

        if( PAPI_OK != (err = PAPI_destroy_eventset(&event_cb->papi_eventset)) ) {
            parsec_debug_verbose(3, parsec_debug_output, "Failed to destroy PAPI eventset (error %s)", PAPI_strerror(err));
        }
        event_cb->papi_eventset = PAPI_NULL;
    }
}
