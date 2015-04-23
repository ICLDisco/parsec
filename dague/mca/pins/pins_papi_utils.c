/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#if defined(HAVE_PAPI)
#include <papi.h>
#endif
#include "pins_papi_utils.h"
#include "dague/utils/output.h"

#if defined(HAVE_PAPI)
static int init_done = 0;
static int thread_init_done = 0;
#endif  /* defined(HAVE_PAPI) */

void pins_papi_init(dague_context_t * master_context)
{
    (void)master_context;
#if defined(HAVE_PAPI)
    if (!init_done) {
        init_done = 1;
        PAPI_library_init(PAPI_VER_CURRENT); /* this has to happen before threads get created */
        PAPI_set_debug(PAPI_VERB_ECONT);
        int t_init = PAPI_thread_init(( unsigned long ( * )( void ) ) ( pthread_self )); 
        if (t_init != PAPI_OK)
            DEBUG(("PAPI Thread Init failed with error code %d (%s)!\n", t_init, PAPI_strerror(t_init)));
    }
#endif /* HAVE_PAPI */
}


void pins_papi_thread_init(dague_execution_unit_t * exec_unit)
{
    (void)exec_unit;
#if defined(HAVE_PAPI)
    if (!thread_init_done) {
        thread_init_done = 1;
        int rv = PAPI_register_thread();
        if (rv != PAPI_OK)
            DEBUG(("PAPI_register_thread failed with error %s\n", PAPI_strerror(rv)));
    }
#endif /* HAVE_PAPI */
}

/**
 * Grow the events array to the expected number. Resize the internal array, and
 * copy the old data into the new position. Can be used to create the initial set
 * as long as the correct value for num_events (0) and for the pointer to the
 * array of events (NULL) are set into the events structure.
 *
 * Returns: 0 if the events has been succesfully grown.
 *         -1 in all other cases. The events have been left untouched.
 */
static int grow_events(parsec_pins_papi_events_t* events, int size)
{
    void* tmp;
    int i;

    assert(size > events->num_counters);
    tmp = realloc(events->events, size * sizeof(parsec_pins_papi_event_t));
    if( NULL == tmp )  /* realloc failed */
        return -1;
    events->events = (parsec_pins_papi_event_t*)tmp;

    for(i = events->num_counters; i < size; i++) {
        events->events[i].pins_papi_event_name = NULL;
        events->events[i].pins_papi_native_event = PAPI_NULL;
        events->events[i].socket = -1;
        events->events[i].core = -1;
        events->events[i].frequency = 1;
    }
    events->num_allocated_counters = size;
    return 0;
}

parsec_pins_papi_events_t* parsec_pins_papi_events_new(char* events_str)
{
    char *mca_param_name, *token, *save_hptr = NULL, *save_lptr = NULL;
    int err, i, socket, core, tmp_eventset = PAPI_NULL;
    parsec_pins_papi_events_t* events = (parsec_pins_papi_events_t*)malloc(sizeof(parsec_pins_papi_events_t));
    parsec_pins_papi_event_t* event;

    events->num_counters = 0;
    events->events = NULL;
    grow_events(events, NUM_DEFAULT_EVENTS);

    mca_param_name = strdup(events_str);
    token = strtok_r(mca_param_name, ",", &save_hptr);

    if( PAPI_OK != (err = PAPI_create_eventset(&tmp_eventset)) ) {
        dague_output(0, "%s: couldn't create the PAPI event set; ERROR: %s\n",
                     __func__, PAPI_strerror(err));
        return NULL;
    }

    while(token != NULL) {
        if( events->num_counters == events->num_allocated_counters ) {
            grow_events(events, 2 * events->num_allocated_counters);
        }
        event = &events->events[events->num_counters];

        for( token = strtok_r(token, ":", &save_lptr); NULL != token; token = strtok_r(NULL, ":", &save_lptr) ) {
            if(token[0] == 'S') {
                if(token[1] != '*') {
                    event->socket = atoi(&token[1]);
                }
                continue;
            }

            if(token[0] == 'C') {
                if(token[1] != '*') {
                    event->core = atoi(&token[1]);
                }
                continue;
            }

            if(token[0] == 'F') {
                event->frequency = atoi(&token[1]);
                continue;
            }
            if( event->frequency <= 0 ) {
                dague_output(0, "%s: Unsupported frequency (%d must be > 0). Discard the event.\n",
                             __func__, token);
                break;
            }

            /* Convert event name to code */
            if(PAPI_OK != (err = PAPI_event_name_to_code(token, &event->pins_papi_native_event)) ) {
                dague_output(0, "%s: Could not convert %s to a valid PAPI event name (%s). Ignore the event\n",
                             __func__, token, PAPI_strerror(err));
            } else {
                if( PAPI_OK != (err = PAPI_add_event(tmp_eventset,
                                                     event->pins_papi_native_event)) ) {
                    dague_output(0, "%s: Unsupported event %s [%x](ERROR: %s). Discard the event.\n",
                                 __func__, token, event->pins_papi_native_event, PAPI_strerror(err));
                    continue;
                }
                dague_output(0, "Valid PAPI event %s on %d socket (-1 for all), %d core (-1 for all) and frequency %d\n",
                             token, event->socket, event->core, event->frequency);
                /* We have a valid event, let's move to the next */
                event->pins_papi_event_name = strdup(token);
                events->num_counters++;
            }
            break;
        }
        token = strtok_r(NULL, ",", &save_hptr);
    }

    free(mca_param_name);
    if( PAPI_NULL != tmp_eventset ) {
        (void)PAPI_cleanup_eventset(tmp_eventset);  /* just do it and don't complain */
    }
    return events;
}

int parsec_pins_papi_events_free(parsec_pins_papi_events_t** pevents)
{
    parsec_pins_papi_events_t* events = *pevents;
    parsec_pins_papi_event_t* event;
    int i;

    for( i = 0; i < events->num_counters; i++ ) {
        event = &events->events[i];
        if( NULL != event->pins_papi_event_name )
            free(event->pins_papi_event_name);
    }
    free(events->events); events->events = NULL;
    free(events);
    *pevents = NULL;
    return 0;
}
