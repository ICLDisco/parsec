/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#include "dague_config.h"
#include "pins_papi_core.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "dague/utils/output.h"
#include "dague/utils/mca_param.h"
#include <stdio.h>
#include <papi.h>
#include "execution_unit.h"

typedef struct parsec_pins_core_callback_s {
    parsec_pins_next_callback_t  default_cb;
    int                          papi_eventset;
    int                          num_counters;
    char**                       pins_papi_event_name;
    int*                         pins_papi_native_event;
    int                          pins_prof_event[2];
    int                          num_tasks;
    int                          begin_end;
} parsec_pins_core_callback_t;

static void start_papi_core(dague_execution_unit_t * exec_unit,
                            dague_execution_context_t * exec_context,
                            parsec_pins_next_callback_t* data);

static void stop_papi_core(dague_execution_unit_t * exec_unit,
                           dague_execution_context_t * exec_context,
                           parsec_pins_next_callback_t* data);

static char* mca_param_string;

static void pins_cleanup_event(parsec_pins_core_callback_t* event_cb,
                               papi_core_info_t* pinfo)
{
    int i, err;

    if(PAPI_NULL != event_cb->papi_eventset) {
        if( PAPI_OK != (err = PAPI_stop(event_cb->papi_eventset, pinfo->values)) ) {
            dague_output(0, "couldn't stop PAPI eventset ERROR: %s\n",
                         PAPI_strerror(err));
        }
        /* the counting should be stopped by now */
        for(i = 0; i < event_cb->num_counters; i++) {
            if( PAPI_OK != (err = PAPI_remove_event(event_cb->papi_eventset,
                                                    event_cb->pins_papi_native_event[i])) ) {
                dague_output(0, "failed to remove event %s; ERROR: %s\n",
                             event_cb->pins_papi_event_name[i], PAPI_strerror(err));
            }
        }
        if( PAPI_OK != (err = PAPI_cleanup_eventset(event_cb->papi_eventset)) ) {
            dague_output(0, "failed to cleanup eventset (ERROR: %s)\n", PAPI_strerror(err));
        }

        if( PAPI_OK != (err = PAPI_destroy_eventset(&event_cb->papi_eventset)) ) {
            dague_output(0, "failed to destroy PAPI eventset (ERROR: %s)\n", PAPI_strerror(err));
        }
    }

    for(i = 0; i < event_cb->num_counters; i++)
        if( NULL != event_cb->pins_papi_event_name[i] )
        free(event_cb->pins_papi_event_name[i]);

    if( NULL != event_cb->pins_papi_event_name )
        free(event_cb->pins_papi_event_name);
    if( NULL != event_cb->pins_papi_native_event)
        free(event_cb->pins_papi_native_event);
}

static void pins_init_papi_core(dague_context_t * master_context)
{
    pins_papi_init(master_context);

    dague_mca_param_reg_string_name("pins", "core_event",
                                    "PAPI event to be saved.\n",
                                    false, false,
                                    "", &mca_param_string);
}

static void pins_fini_papi_core(dague_context_t * master_context)
{
}

static void pins_thread_init_papi_core(dague_execution_unit_t * exec_unit)
{
    char *mca_param_name, *token, *saveptr = NULL;
    int err, i;
    bool socket, core, started = false;
    parsec_pins_core_callback_t* event_cb = (parsec_pins_core_callback_t*)malloc(sizeof(parsec_pins_core_callback_t));

    event_cb->num_counters = 0;
    event_cb->pins_papi_event_name = (char**)calloc(NUM_CORE_EVENTS, sizeof(char*));
    event_cb->pins_papi_native_event = (int*)calloc(NUM_CORE_EVENTS, sizeof(int));
    event_cb->papi_eventset = PAPI_NULL;

    for(i = 0; i < NUM_CORE_EVENTS; i++) {
        event_cb->pins_papi_event_name[i] = NULL;
        event_cb->pins_papi_native_event[i] = PAPI_NULL;
    }

    mca_param_name = strdup(mca_param_string);
    token = strtok_r(mca_param_name, ":", &saveptr);

    while(token != NULL) {
        socket = core = false;

        if(token[0] == 'S') {
            if(token[1] != '*') {
                if(atoi(&token[1]) == exec_unit->socket_id)
                    socket = true;
            } else
                socket = true;
        }

        token = strtok_r(NULL, ":", &saveptr);

        if(token[0] == 'C') {
            if(token[1] != '*') {
                if(atoi(&token[1]) == (exec_unit->core_id % CORES_PER_SOCKET))
                    core = true;
            } else
                core = true;
        }

        token = strtok_r(NULL, ",", &saveptr);

        if(socket && core) {
            if(event_cb->num_counters == NUM_CORE_EVENTS) {
                dague_output(0, "pins_thread_init_papi_core: thread %d couldn't add event '%s' because only %d events are allowed.\n",
                             exec_unit->th_id, token, NUM_CORE_EVENTS);
                break;
            }

            /* Convert event name to code */
            if(PAPI_OK != PAPI_event_name_to_code(token, &event_cb->pins_papi_native_event[event_cb->num_counters]) )
                break;

            if(!started) {
                /* Create an empty eventset */
                if( PAPI_OK != (err = PAPI_create_eventset(&event_cb->papi_eventset)) ) {
                    dague_output(0, "pins_thread_init_papi_core: thread %d couldn't create the PAPI event set; ERROR: %s\n",
                                 exec_unit->th_id, PAPI_strerror(err));
                    break;
                }
                started = true;
            }

            event_cb->pins_papi_event_name[event_cb->num_counters] = strdup(token);

            /* Add events to the eventset */
            if( PAPI_OK != (err = PAPI_add_event(event_cb->papi_eventset,
                                                 event_cb->pins_papi_native_event[event_cb->num_counters])) ) {
                dague_output(0, "pins_thread_init_papi_core: failed to add event %s; ERROR: %s\n",
                             token, PAPI_strerror(err));
                break;
            }
            event_cb->num_counters++;
        }
        token = strtok_r(NULL, ":", &saveptr);
    }

    free(mca_param_name);

    if(event_cb->num_counters > 0) {
        char* key_string;
        char* value_string;
        int string_size = 0;

        asprintf(&key_string, "PINS_CORE_S%d_C%d", exec_unit->socket_id, exec_unit->core_id);

        for(i = 0; i < event_cb->num_counters; i++) {
            string_size += strlen(event_cb->pins_papi_event_name[i]) + strlen("{int64_t}"PARSEC_PINS_SEPARATOR);
        }

        value_string = (char*)calloc(string_size, sizeof(char));

        for(i = 0; i < event_cb->num_counters; i++) {
            strcat(value_string, event_cb->pins_papi_event_name[i]);
            strcat(value_string, "{int64_t}"PARSEC_PINS_SEPARATOR);
        }

        dague_profiling_add_dictionary_keyword(key_string, "fill:#00AAFF",
                                               sizeof(uint64_t) * event_cb->num_counters, value_string,
                                               &event_cb->pins_prof_event[0],
                                               &event_cb->pins_prof_event[1]);
        free(key_string);
        free(value_string);

        if( PAPI_OK != (err = PAPI_start(event_cb->papi_eventset)) ) {
            dague_output(0, "couldn't start PAPI eventset for thread %d; ERROR: %s\n",
                         exec_unit->th_id, PAPI_strerror(err));
        }
        PINS_REGISTER(exec_unit, EXEC_BEGIN, start_papi_core,
                      (parsec_pins_next_callback_t*)event_cb);
        PINS_REGISTER(exec_unit, EXEC_END, stop_papi_core,
                      (parsec_pins_next_callback_t*)event_cb);
        return;  /* we're done here */
    }
    papi_core_info_t info;
    pins_cleanup_event(event_cb, &info);
    free(event_cb);
}

static void pins_thread_fini_papi_core(dague_execution_unit_t * exec_unit)
{
    parsec_pins_core_callback_t* event_cb;
    papi_core_info_t info;

    PINS_UNREGISTER(exec_unit, EXEC_BEGIN, start_papi_core, (parsec_pins_next_callback_t**)&event_cb);
    PINS_UNREGISTER(exec_unit, EXEC_END, stop_papi_core, (parsec_pins_next_callback_t**)&event_cb);

    if( (NULL == event_cb) || (PAPI_NULL == event_cb->papi_eventset) )
        return;

    pins_cleanup_event(event_cb, &info);
    /* If the last profiling event was an 'end' event */
    if(event_cb->begin_end == 0) {
        (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_event[0],
                                    45, 0, (void *)&info);
    }
    (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_event[1],
                                45, 0, (void *)&info);
    free(event_cb);
}

static void start_papi_core(dague_execution_unit_t* exec_unit,
                            dague_execution_context_t* exec_context,
                            parsec_pins_next_callback_t* cb_data)
{
    parsec_pins_core_callback_t* event_cb = (parsec_pins_core_callback_t*)cb_data;
    papi_core_info_t info;
    int err;

    if( PAPI_NULL == event_cb->papi_eventset )
        return;

    if( PAPI_OK != (err = PAPI_read(event_cb->papi_eventset, info.values)) ) {
        dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        return;
    }

    (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_event[0],
                                45, 0, (void *)&info);
}

static void stop_papi_core(dague_execution_unit_t* exec_unit,
                           dague_execution_context_t* exec_context,
                           parsec_pins_next_callback_t* cb_data)
{
    parsec_pins_core_callback_t* event_cb = (parsec_pins_core_callback_t*)cb_data;
    papi_core_info_t info;
    int err;

    if( PAPI_NULL == event_cb->papi_eventset )
        return;

    if( PAPI_OK != (err = PAPI_read(event_cb->papi_eventset, info.values)) ) {
        dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        return;
    }
    (void)dague_profiling_trace(exec_unit->eu_profile, event_cb->pins_prof_event[1],
                                45, 0, (void *)&info);
}

const dague_pins_module_t dague_pins_papi_core_module = {
    &dague_pins_papi_core_component,
    {
        pins_init_papi_core,
        pins_fini_papi_core,
        NULL,
        NULL,
        pins_thread_init_papi_core,
        pins_thread_fini_papi_core,
    }
};
