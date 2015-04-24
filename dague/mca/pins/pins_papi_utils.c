/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include <papi.h>
#include "pins_papi_utils.h"
#include "dague/utils/output.h"

static int init_done = 0;

/**
 * This function should be called once per application in order to enable
 * a correct usage of PAPI.
 */
int pins_papi_init(dague_context_t * master_context)
{
    int err;

    (void)master_context;
    if( !init_done ) {
        init_done = 1;
        err = PAPI_library_init(PAPI_VER_CURRENT); /* this has to happen before threads get created */
        if( PAPI_VER_CURRENT != err ) {
            dague_output(0, "Failed to initialize PAPI (%x != %x[expected]). All components depending on PAPI will be disabled.",
                         PAPI_VER_CURRENT, err);
            return -1;
        }
        dague_output(0, "Using PAPI version %x\n", PAPI_VER_CURRENT);
        /*PAPI_set_debug(PAPI_VERB_ECONT);*/
        err = PAPI_thread_init(( unsigned long ( * )( void ) ) ( pthread_self ));
        if( err != PAPI_OK ) {
            dague_output(0, "PAPI_thread_init failed (%s)! All components depending on PAPI will be disabled.\n", PAPI_strerror(err));
            return -2;
        }
    }
    return 0;
}

/**
 * finalization function to be called once per application. It is the
 * counterpart fo pins_papi_init.
 */
int pins_papi_fini(dague_context_t * master_context)
{
    return 0;
}

/**
 * This function should be called by each thread in order to allow PAPI
 * to know about each of the potential users.
 */
int pins_papi_thread_init(dague_execution_unit_t * exec_unit)
{
    int err;

    (void)exec_unit;
    err = PAPI_register_thread();
    if( err != PAPI_OK ) {
        dague_output(0, "PAPI_register_thread failed (%s). All components depending on PAPI will be disabled.\n", PAPI_strerror(err));
        return -1;
    }
    return 0;
}

/**
 * Function to be called once per thread, similar to pins_papi_thread_init.
 */

int pins_papi_thread_fini(dague_execution_unit_t * exec_unit)
{
    int err = PAPI_unregister_thread();
    if ( err != PAPI_OK )
        dague_output(0, "PAPI_unregister_thread failed (%s).\n", PAPI_strerror(err));
    return 0;
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

        save_lptr = NULL;
        for( ; NULL != token;
             strtok_r((NULL == save_lptr ? token : NULL), ":", &save_lptr), token = save_lptr ) {

            if(token[0] == 'S') {
                if(token[1] != '*') {
                    event->socket = atoi(&token[1]);
                }
                strtok_r((NULL == save_lptr ? token : NULL), ":", &save_lptr);
                token = save_lptr;
                continue;
            }

            if(token[0] == 'C') {
                if(token[1] != '*') {
                    event->core = atoi(&token[1]);
                }
                strtok_r((NULL == save_lptr ? token : NULL), ":", &save_lptr);
                token = save_lptr;
                continue;
            }

            if(token[0] == 'F') {
                event->frequency = atoi(&token[1]);
                strtok_r((NULL == save_lptr ? token : NULL), ":", &save_lptr);
                token = save_lptr;
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

void parsec_pins_papi_event_cleanup(parsec_pins_papi_callback_t* event_cb,
                                    parsec_pins_papi_values_t* pinfo)
{
    int i, err;

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
