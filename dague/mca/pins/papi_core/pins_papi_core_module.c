#include "pins_papi_core.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "dague/utils/output.h"
#include "dague/utils/mca_param.h"
#include <stdio.h>
#include <papi.h>
#include "execution_unit.h"

#define PARSEC_PAPI_SEPARATOR ";"

static void start_papi_core(dague_execution_unit_t * exec_unit,
                            dague_execution_context_t * exec_context,
                            void * data);

static void stop_papi_core(dague_execution_unit_t * exec_unit,
                           dague_execution_context_t * exec_context,
                           void * data);

/* Courtesy calls to previously-registered cbs */
static parsec_pins_callback * exec_begin_prev;
static parsec_pins_callback * exec_end_prev;

static char* mca_param_string;

static void pins_init_papi_core(dague_context_t * master_context) {
    pins_papi_init(master_context);

    dague_mca_param_reg_string_name("pins", "core_event",
                                    "PAPI event to be saved.\n",
                                    false, false,
                                    "", &mca_param_string);

    /* prepare link to the previously registered PINS module */
    exec_begin_prev = PINS_REGISTER(EXEC_BEGIN, start_papi_core);
    exec_end_prev = PINS_REGISTER(EXEC_END, stop_papi_core);
}

static void pins_fini_papi_core(dague_context_t * master_context) {
    (void) master_context;
    PINS_REGISTER(EXEC_BEGIN, exec_begin_prev);
    PINS_REGISTER(EXEC_END, exec_end_prev);
}

static void pins_thread_init_papi_core(dague_execution_unit_t * exec_unit) {
    char *mca_param_name, *token, *saveptr = NULL;
    int err, i;
    bool socket, core, started = false;

    exec_unit->num_core_counters = 0;
    exec_unit->pins_papi_core_event_name = (char**)calloc(NUM_CORE_EVENTS, sizeof(char*));
    exec_unit->pins_papi_core_native_event = (int*)calloc(NUM_CORE_EVENTS, sizeof(int));
    exec_unit->papi_eventsets[PER_CORE_SET] = PAPI_NULL;

    for(i = 0; i < NUM_CORE_EVENTS; i++) {
        exec_unit->pins_papi_core_event_name[i] = NULL;
        exec_unit->pins_papi_core_native_event[i] = PAPI_NULL;
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
            if(exec_unit->num_core_counters == NUM_CORE_EVENTS) {
                dague_output(0, "pins_thread_init_papi_core: thread %d couldn't add event '%s' because only %d events are allowed.\n",
                             exec_unit->th_id, token, NUM_CORE_EVENTS);
                break;
            }

            /* Convert event name to code */
            if(PAPI_OK != PAPI_event_name_to_code(token, &exec_unit->pins_papi_core_native_event[exec_unit->num_core_counters]) )
                break;

            if(!started) {
                /* Create an empty eventset */
                if( PAPI_OK != (err = PAPI_create_eventset(&exec_unit->papi_eventsets[PER_CORE_SET])) ) {
                    dague_output(0, "pins_thread_init_papi_core: thread %d couldn't create the PAPI event set; ERROR: %s\n",
                                 exec_unit->th_id, PAPI_strerror(err));
                    break;
                }
                started = true;
            }

            exec_unit->pins_papi_core_event_name[exec_unit->num_core_counters] = strdup(token);

            /* Add events to the eventset */
            if( PAPI_OK != (err = PAPI_add_event(exec_unit->papi_eventsets[PER_CORE_SET],
                                                 exec_unit->pins_papi_core_native_event[exec_unit->num_core_counters])) ) {
                dague_output(0, "pins_thread_init_papi_core: failed to add event %s; ERROR: %s\n",
                             token, PAPI_strerror(err));
                break;
            }
            exec_unit->num_core_counters++;
        }
        token = strtok_r(NULL, ":", &saveptr);
    }

    free(mca_param_name);

    if(exec_unit->num_core_counters > 0) {
        char* key_string;
        char* value_string;
        int string_size = 0;

        asprintf(&key_string, "PINS_CORE_S%d_C%d", exec_unit->socket_id, exec_unit->core_id);

        for(i = 0; i < exec_unit->num_core_counters; i++) {
            string_size += strlen(exec_unit->pins_papi_core_event_name[i]) + strlen("{int64_t}"PARSEC_PAPI_SEPARATOR);
        }

        value_string = (char*)calloc(string_size, sizeof(char));

        for(i = 0; i < exec_unit->num_core_counters; i++) {
            strcat(value_string, exec_unit->pins_papi_core_event_name[i]);
            strcat(value_string, "{int64_t}"PARSEC_PAPI_SEPARATOR);
        }

        dague_profiling_add_dictionary_keyword(key_string, "fill:#00AAFF",
                                               sizeof(uint64_t) * exec_unit->num_core_counters, value_string,
                                               &exec_unit->pins_prof_papi_core[0],
                                               &exec_unit->pins_prof_papi_core[1]);
        free(key_string);
        free(value_string);

        if( PAPI_OK != (err = PAPI_start(exec_unit->papi_eventsets[PER_CORE_SET])) ) {
            dague_output(0, "couldn't start PAPI eventset for thread %d; ERROR: %s\n",
                         exec_unit->th_id, PAPI_strerror(err));
        }
    }
}

static void pins_thread_fini_papi_core(dague_execution_unit_t * exec_unit) {
    int err, i;

    if( PAPI_NULL == exec_unit->papi_eventsets[PER_CORE_SET] )
        return;

    papi_core_info_t info;
    if( PAPI_OK != (err = PAPI_stop(exec_unit->papi_eventsets[PER_CORE_SET], info.values)) ) {
        dague_output(0, "couldn't stop PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
    }
    /* the counting should be stopped by now */
    for(i = 0; i < exec_unit->num_core_counters; i++) {
        if( PAPI_OK != (err = PAPI_remove_event(exec_unit->papi_eventsets[PER_CORE_SET],
                                                exec_unit->pins_papi_core_native_event[i])) ) {
            dague_output(0, "pins_thread_fini_papi_core: failed to remove event %s; ERROR: %s\n",
                         exec_unit->pins_papi_core_event_name[i], PAPI_strerror(err));
        }
    }

    for(i = 0; i < exec_unit->num_core_counters; i++)
        free(exec_unit->pins_papi_core_event_name[i]);

    free(exec_unit->pins_papi_core_event_name);
    free(exec_unit->pins_papi_core_native_event);

    if( PAPI_OK != (err = PAPI_cleanup_eventset(exec_unit->papi_eventsets[PER_CORE_SET])) ) {
        dague_output(0, "pins_thread_fini_papi_core: failed to cleanup thread %d eventset; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
    }
    if( PAPI_OK != (err = PAPI_destroy_eventset(&exec_unit->papi_eventsets[PER_CORE_SET])) ) {
        dague_output(0, "pins_thread_fini_papi_core: failed to destroy thread %d eventset; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
    }
}

static void start_papi_core(dague_execution_unit_t * exec_unit,
                            dague_execution_context_t * exec_context,
                            void * data)
{
    if( PAPI_NULL == exec_unit->papi_eventsets[PER_CORE_SET] )
        goto next_pins;

    papi_core_info_t info;
    int err;

    if( PAPI_OK != (err = PAPI_read(exec_unit->papi_eventsets[PER_CORE_SET], info.values)) ) {
        dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        goto next_pins;
    }

    (void)dague_profiling_trace(exec_unit->eu_profile, exec_unit->pins_prof_papi_core[0],
                                45, 0, (void *)&info);

  next_pins:
    /* call previous callback, if any */
    if (NULL != exec_begin_prev)
        (*exec_begin_prev)(exec_unit, exec_context, data);

    (void)exec_context; (void)data;
}

static void stop_papi_core(dague_execution_unit_t * exec_unit,
                           dague_execution_context_t * exec_context,
                           void * data) {
    if( PAPI_NULL == exec_unit->papi_eventsets[PER_CORE_SET] )
        goto next_pins;

    papi_core_info_t info;
    int err;

    if( PAPI_OK != (err = PAPI_read(exec_unit->papi_eventsets[PER_CORE_SET], info.values)) ) {
        dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        goto next_pins;
    }
    printf("Core %d: %lld   %lld\n", exec_unit->core_id, info.values[0], info.values[1]);
    (void)dague_profiling_trace(exec_unit->eu_profile, exec_unit->pins_prof_papi_core[1],
                                45, 0, (void *)&info);

  next_pins:
    /* call previous callback, if any */
    if (NULL != exec_end_prev)
        (*exec_end_prev)(exec_unit, exec_context, data);

    (void)exec_context; (void)data;
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
