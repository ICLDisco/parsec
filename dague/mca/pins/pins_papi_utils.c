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
static int init_status = DAGUE_SUCCESS;

/**
 * This function should be called once per application in order to enable
 * a correct usage of PAPI.
 */
int pins_papi_init(dague_context_t * master_context)
{
    int err;

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
    char *mca_param_name, *token, *save_hptr = NULL, *save_lptr = NULL;
    int err, i, socket, core, tmp_eventset = PAPI_NULL;
    parsec_pins_papi_events_t* events = (parsec_pins_papi_events_t*)malloc(sizeof(parsec_pins_papi_events_t));
    parsec_pins_papi_event_t* event;

    events->num_counters = 0;
    events->events = NULL;

    mca_param_name = strdup(events_str);
    token = strtok_r(mca_param_name, ",", &save_hptr);

    if( PAPI_OK != (err = PAPI_create_eventset(&tmp_eventset)) ) {
        dague_output(0, "%s: couldn't create the PAPI event set; ERROR: %s\n",
                     __func__, PAPI_strerror(err));
        return NULL;
    }

    while(token != NULL) {

        save_lptr = NULL;
        for( token = strtok_r(token, ":", &save_lptr);
             NULL != token;
             token = strtok_r(NULL, ":", &save_lptr) ) {

            /* Handle the event creation. If the event is not NULL then we inherited a failed event
             * from a previous iteration. So, clean it up and use it as is.
             */
            if( NULL == event ) {
                event = calloc( 1, sizeof(parsec_pins_papi_event_t) );
            } else {
                memset( event, 0, sizeof(parsec_pins_papi_event_t) );
            }
            event->socket = -1;
            event->core = -1;
            event->frequency = 1;

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
                break;
            } else {
                PAPI_event_info_t papi_info;

                if( PAPI_OK != (err = PAPI_add_event(tmp_eventset,
                                                     event->pins_papi_native_event)) ) {
                    dague_output(0, "%s: Unsupported event %s [%x](ERROR: %s). Discard the event.\n",
                                 __func__, token, event->pins_papi_native_event, PAPI_strerror(err));
                    break;
                }
                dague_output(0, "Valid PAPI event %s on socket %d (-1 for all), core %d (-1 for all) with frequency %d\n",
                             token, event->socket, event->core, event->frequency);
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
            }
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
