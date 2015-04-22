/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "pins_papi_socket.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "dague/utils/output.h"
#include "dague/utils/mca_param.h"
#include <stdio.h>
#include <papi.h>
#include "execution_unit.h"

typedef struct parsec_pins_socket_callback_s {
    parsec_pins_next_callback_t  default_cb;
    int                          papi_eventset;
    int                          num_socket_counters;
    char**                       pins_papi_socket_event_name;
    int*                         pins_papi_socket_native_event;
    int                          pins_prof_papi_socket[2];
    int                          num_socket_tasks;
    int                          begin_end;
    int                          num_tasks;
} parsec_pins_socket_callback_t;

static void stop_papi_socket(dague_execution_unit_t* exec_unit,
                             dague_execution_context_t* exec_context,
                             parsec_pins_next_callback_t* cb_data);

static char* mca_param_string;

static void pins_init_papi_socket(dague_context_t * master_context)
{
    pins_papi_init(master_context);

    dague_mca_param_reg_string_name("pins", "socket_event",
                                    "PAPI event to be saved.\n",
                                    false, false,
                                    "", &mca_param_string);
}

static void pins_thread_init_papi_socket(dague_execution_unit_t * exec_unit)
{
    char* mca_param_name, *token, *temp, *saveptr = NULL;
    int err, i;
    bool socket, core, started = false;
    parsec_pins_socket_callback_t* event_cb = (parsec_pins_socket_callback_t*)malloc(sizeof(parsec_pins_socket_callback_t));

    event_cb->num_socket_counters = 0;
    event_cb->num_socket_tasks = 0;
    event_cb->begin_end = 0;
    event_cb->pins_papi_socket_event_name = (char**)calloc(NUM_SOCKET_EVENTS, sizeof(char*));
    event_cb->pins_papi_socket_native_event = (int*)calloc(NUM_SOCKET_EVENTS, sizeof(int));
    event_cb->papi_eventset = PAPI_NULL;

    for(i = 0; i < NUM_SOCKET_EVENTS; i++) {
        event_cb->pins_papi_socket_event_name[i] = NULL;
        event_cb->pins_papi_socket_native_event[i] = PAPI_NULL;
    }

    mca_param_name = strdup(mca_param_string);
    token = strtok_r(mca_param_name, ":", &saveptr);

    if(token == NULL) {
        dague_output(0, "No PAPI events have been specified.  None will be recorded.\n");
        event_cb->papi_eventset = PAPI_NULL;
        return;
    }

    while(token != NULL) {
        socket = core = false;

        if(token[0] == 'S') {
            temp = (char*)calloc(strlen(token), sizeof(char));
            strcpy(temp, token);
            memmove(temp, temp+1, strlen(temp));

            if(temp[0] != '*') {
                if(atoi(temp) == exec_unit->socket_id)
                    socket = true;
            } else
                socket = true;
            free(temp);
        }

        token = strtok_r(NULL, ":", &saveptr);

        if(token[0] == 'C') {
            temp = (char*)calloc(strlen(token),sizeof(char));
            strcpy(temp, token);
            memmove(temp, temp+1, strlen(temp));

            if(temp[0] != '*') {
                if(atoi(temp) == (exec_unit->core_id % CORES_PER_SOCKET))
                    core = true;
            } else
                core = true;
            free(temp);
        }

        token = strtok_r(NULL, ",", &saveptr);

        if(socket && core) {
            if(event_cb->num_socket_counters == NUM_SOCKET_EVENTS) {
                dague_output(0, "pins_thread_init_papi_socket: thread %d couldn't add event '%s' because only %d events are allowed.\n",
                             exec_unit->th_id, token, NUM_SOCKET_EVENTS);
                break;
            }

            if(!started) {
                pins_papi_thread_init(exec_unit);
                event_cb->papi_eventset = PAPI_NULL;

                /* Create an empty eventset */
                if( PAPI_OK != (err = PAPI_create_eventset(&event_cb->papi_eventset)) ) {
                    dague_output(0, "pins_thread_init_papi_socket: thread %d couldn't create the PAPI event set; ERROR: %s\n",
                                 exec_unit->th_id, PAPI_strerror(err));
                    return;
                }
                started = true;
            }

            /* Convert event name to code */
            if(PAPI_OK == PAPI_event_name_to_code(token, &event_cb->pins_papi_socket_native_event[event_cb->num_socket_counters]) )
                event_cb->pins_papi_socket_event_name[event_cb->num_socket_counters] = strdup(token);

            if(PAPI_NULL == event_cb->pins_papi_socket_native_event[event_cb->num_socket_counters]) {
                dague_output(0, "No event derived from %s is supported on this system (use papi_native_avail for a complete list)\n", token);
                return;
            }

            /* Add events to the eventset */
            if( PAPI_OK != (err = PAPI_add_event(event_cb->papi_eventset,
                                                 event_cb->pins_papi_socket_native_event[event_cb->num_socket_counters])) ) {
                dague_output(0, "pins_thread_init_papi_socket: failed to add event %s; ERROR: %s\n",
                             token, PAPI_strerror(err));
                return;
            }
            event_cb->num_socket_counters++;
        }
        token = strtok_r(NULL, ":", &saveptr);
    }

    free(mca_param_name);
    free(token);

    if(event_cb->num_socket_counters > 0) {
        papi_socket_info_t info;
        char* key_string;
        char* value_string;
        int string_size = 0;

        asprintf(&key_string, "PINS_SOCKET_S%d_C%d", exec_unit->socket_id, exec_unit->core_id);

        for(i = 0; i < event_cb->num_socket_counters; i++) {
            string_size += strlen(event_cb->pins_papi_socket_event_name[i]) + strlen("{int64_t}"PARSEC_PINS_SEPARATOR);
        }

        value_string = (char*)calloc(string_size, sizeof(char));

        for(i = 0; i < event_cb->num_socket_counters; i++) {
            strcat(value_string, event_cb->pins_papi_socket_event_name[i]);
            strcat(value_string, "{int64_t}"PARSEC_PINS_SEPARATOR);
        }

        dague_profiling_add_dictionary_keyword(key_string, "fill:#00AAFF",
                                               sizeof(papi_socket_info_t), value_string,
                                               &event_cb->pins_prof_papi_socket[0],
                                               &event_cb->pins_prof_papi_socket[1]);
        free(key_string);
        free(value_string);
        /* Start the PAPI counters. */
        if( PAPI_OK != (err = PAPI_start(event_cb->papi_eventset)) ) {
            dague_output(0, "couldn't start PAPI eventset for thread %d; ERROR: %s\n",
                         exec_unit->th_id, PAPI_strerror(err));
            return;
        }

        if( PAPI_OK != (err = PAPI_read(event_cb->papi_eventset, info.values)) ) {
            dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                         exec_unit->th_id, PAPI_strerror(err));
            return;
        }

        (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_papi_socket[event_cb->begin_end],
                                    45, 0, (void *)&info);
        event_cb->begin_end = (event_cb->begin_end + 1) & 0x1;  /* aka. % 2 */
        PINS_REGISTER(exec_unit, EXEC_END, stop_papi_socket, (parsec_pins_next_callback_t*)event_cb);
    }
}

static void pins_thread_fini_papi_socket(dague_execution_unit_t* exec_unit)
{
    parsec_pins_socket_callback_t* event_cb;
    papi_socket_info_t info;
    int err, i;

    PINS_UNREGISTER(exec_unit, EXEC_END, stop_papi_socket, (parsec_pins_next_callback_t**)&event_cb);

    if( PAPI_NULL == event_cb->papi_eventset )
        return;

    /* Stop the PAPI counters. */
    if( PAPI_OK != (err = PAPI_stop(event_cb->papi_eventset, info.values)) ) {
        dague_output(0, "couldn't stop PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
    } else {
        /* If the last profiling event was an 'end' event */
        if(event_cb->begin_end == 0) {
            (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_papi_socket[0],
                                        45, 0, (void *)&info);
        }
        (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_papi_socket[1],
                                    45, 0, (void *)&info);
    }

    /* the counting should be stopped by now */
    for(i = 0; i < event_cb->num_socket_counters; i++) {
        if( PAPI_OK != (err = PAPI_remove_event(event_cb->papi_eventset,
                                                event_cb->pins_papi_socket_native_event[i])) ) {
            dague_output(0, "pins_thread_fini_papi_socket: failed to remove event %s; ERROR: %s\n",
                         event_cb->pins_papi_socket_event_name[i], PAPI_strerror(err));
        }
    }

    for(i = 0; i < event_cb->num_socket_counters; i++)
        free(event_cb->pins_papi_socket_event_name[i]);

    free(event_cb->pins_papi_socket_event_name);
    free(event_cb->pins_papi_socket_native_event);

    if( PAPI_OK != (err = PAPI_cleanup_eventset(event_cb->papi_eventset)) ) {
        dague_output(0, "pins_thread_fini_papi_socket: failed to cleanup thread %d eventset; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
    }
    if( PAPI_OK != (err = PAPI_destroy_eventset(&event_cb->papi_eventset)) ) {
        dague_output(0, "pins_thread_fini_papi_socket: failed to destroy thread %d eventset; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
    }
    free(event_cb);
}

static void stop_papi_socket(dague_execution_unit_t* exec_unit,
                             dague_execution_context_t* exec_context,
                             parsec_pins_next_callback_t* cb_data)
{
    parsec_pins_socket_callback_t* event_cb = (parsec_pins_socket_callback_t*)cb_data;

    if( PAPI_NULL == event_cb->papi_eventset )
        goto next_pins;

    event_cb->num_socket_tasks++;
    if(event_cb->num_socket_tasks == 5) {
        papi_socket_info_t info;
        int err;


        if( PAPI_OK != (err = PAPI_read(event_cb->papi_eventset, info.values)) ) {
            dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                        exec_unit->th_id, PAPI_strerror(err));
            goto next_pins;
        }
        (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_papi_socket[event_cb->begin_end],
                                    45, 0, (void *)&info);
        event_cb->begin_end = (event_cb->begin_end + 1) & 0x1;  /* aka. % 2 */
        event_cb->num_socket_tasks = 0;
    }

    next_pins:

    (void)exec_context;
}

const dague_pins_module_t dague_pins_papi_socket_module = {
    &dague_pins_papi_socket_component,
    {
        pins_init_papi_socket,
        NULL,
        NULL,
        NULL,
        pins_thread_init_papi_socket,
        pins_thread_fini_papi_socket,
    }
};
