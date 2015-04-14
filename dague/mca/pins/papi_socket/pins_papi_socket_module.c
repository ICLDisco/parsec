#include "pins_papi_socket.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "dague/utils/output.h"
#include "dague/utils/mca_param.h"
#include <stdio.h>
#include <papi.h>
#include "execution_unit.h"

static void stop_papi_socket(dague_execution_unit_t * exec_unit,
                 dague_execution_context_t * exec_context,
                 void * data);

/* Courtesy calls to previously-registered cbs */
static parsec_pins_callback * exec_end_prev;

static char* mca_param_string;

static void pins_init_papi_socket(dague_context_t * master_context) {
    pins_papi_init(master_context);

    dague_mca_param_reg_string_name("pins", "socket_event",
                                    "PAPI event to be saved.\n",
                                    false, false,
                                    "", &mca_param_string);

    /* prepare link to the previously registered PINS module */
    exec_end_prev = PINS_REGISTER(EXEC_END, stop_papi_socket);
}

static void pins_fini_papi_socket(dague_context_t * master_context) {
    (void) master_context;
    PINS_REGISTER(EXEC_END, exec_end_prev);
}

static void pins_thread_init_papi_socket(dague_execution_unit_t * exec_unit) {
    int err, i;
    
    exec_unit->papi_eventsets[PER_SOCKET_SET] = PAPI_NULL;

    exec_unit->num_socket_counters = pins_papi_mca_string_parse(exec_unit, mca_param_string, &exec_unit->pins_papi_socket_event_name);
    if(exec_unit->num_socket_counters == 0)
        return;

    if(-1 == pins_papi_create_eventset(exec_unit, &exec_unit->papi_eventsets[PER_SOCKET_SET], exec_unit->pins_papi_socket_event_name,
                                       &exec_unit->pins_papi_socket_native_event, exec_unit->num_socket_counters)) {
        exec_unit->papi_eventsets[PER_SOCKET_SET] = PAPI_NULL;
        return;
    }

	/* Add the dictionary keyword and start the PAPI counters */
    papi_socket_info_t info;
    char* key_string;
    char* value_string;
    int string_size = 0;

    asprintf(&key_string, "PINS_SOCKET_S%d_C%d", exec_unit->socket_id, exec_unit->core_id);

    for(i = 0; i < exec_unit->num_socket_counters; i++)
        string_size += strlen(exec_unit->pins_papi_socket_event_name[i]) + strlen("{int64_t}"PARSEC_PAPI_SEPARATOR);

    value_string = (char*)malloc((string_size+1) * sizeof(char));
    value_string[0] = '\0';
    for(i = 0; i < exec_unit->num_socket_counters; i++) {
        strcat(value_string, exec_unit->pins_papi_socket_event_name[i]);
        strcat(value_string, "{int64_t}"PARSEC_PAPI_SEPARATOR);
    }

    dague_profiling_add_dictionary_keyword(key_string, "fill:#00AAFF",
                                           sizeof(uint64_t) * exec_unit->num_socket_counters, value_string,
                                           &exec_unit->pins_prof_papi_socket[0],
                                           &exec_unit->pins_prof_papi_socket[1]);
    free(key_string);
    free(value_string);
    /* Start the PAPI counters. */
    if( PAPI_OK != (err = PAPI_start(exec_unit->papi_eventsets[PER_SOCKET_SET])) ) {
        dague_output(0, "couldn't start PAPI eventset for thread %d; ERROR: %s\n",
                    exec_unit->th_id, PAPI_strerror(err));
        return;
    }

    if( PAPI_OK != (err = PAPI_read(exec_unit->papi_eventsets[PER_SOCKET_SET], info.values)) ) {
        dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                    exec_unit->th_id, PAPI_strerror(err));
        return;
    }

    (void)dague_profiling_trace(exec_unit->eu_profile, exec_unit->pins_prof_papi_socket[exec_unit->begin_end],
                                45, 0, (void *)&info);
    exec_unit->begin_end = (exec_unit->begin_end + 1) % 2;
}

static void pins_thread_fini_papi_socket(dague_execution_unit_t * exec_unit) {
    int err, i;
    papi_socket_info_t info;

    if( PAPI_NULL == exec_unit->papi_eventsets[PER_SOCKET_SET] )
        return;

    /* Stop the PAPI counters. */
    if( PAPI_OK != (err = PAPI_stop(exec_unit->papi_eventsets[PER_SOCKET_SET], info.values)) ) {
        dague_output(0, "couldn't stop PAPI eventset for thread %d; ERROR: %s\n",
                    exec_unit->th_id, PAPI_strerror(err));
    } else {
        /* If the last profiling event was an 'end' event */
        if(exec_unit->begin_end == 0) {
            (void)dague_profiling_trace(exec_unit->eu_profile, exec_unit->pins_prof_papi_socket[0],
                                        45, 0, (void *)&info);
            (void)dague_profiling_trace(exec_unit->eu_profile, exec_unit->pins_prof_papi_socket[1],
                                        45, 0, (void *)&info);
        } else {
            (void)dague_profiling_trace(exec_unit->eu_profile, exec_unit->pins_prof_papi_socket[1],
                                        45, 0, (void *)&info);
        }
    }

    /* the counting should be stopped by now */
    for(i = 0; i < exec_unit->num_socket_counters; i++) {
        if( PAPI_OK != (err = PAPI_remove_event(exec_unit->papi_eventsets[PER_SOCKET_SET],
                exec_unit->pins_papi_socket_native_event[i])) ) {
            dague_output(0, "pins_thread_fini_papi_socket: failed to remove event %s; ERROR: %s\n",
                        exec_unit->pins_papi_socket_event_name[i], PAPI_strerror(err));
        }
    }

    for(i = 0; i < exec_unit->num_socket_counters; i++)
        free(exec_unit->pins_papi_socket_event_name[i]);

    free(exec_unit->pins_papi_socket_event_name);
    free(exec_unit->pins_papi_socket_native_event);

    if( PAPI_OK != (err = PAPI_cleanup_eventset(exec_unit->papi_eventsets[PER_SOCKET_SET])) ) {
        dague_output(0, "pins_thread_fini_papi_socket: failed to cleanup thread %d eventset; ERROR: %s\n",
                    exec_unit->th_id, PAPI_strerror(err));
    }
    if( PAPI_OK != (err = PAPI_destroy_eventset(&exec_unit->papi_eventsets[PER_SOCKET_SET])) ) {
        dague_output(0, "pins_thread_fini_papi_socket: failed to destroy thread %d eventset; ERROR: %s\n",
                    exec_unit->th_id, PAPI_strerror(err));
    }
}

static void stop_papi_socket(dague_execution_unit_t * exec_unit,
                 dague_execution_context_t * exec_context,
                 void * data) {
    if( PAPI_NULL == exec_unit->papi_eventsets[PER_SOCKET_SET] )
        goto next_pins;

    exec_unit->num_socket_tasks++;
    if(exec_unit->num_socket_tasks == 5) {
        papi_socket_info_t info;
        int err;

        exec_unit->num_socket_tasks = 0;

        if( PAPI_OK != (err = PAPI_read(exec_unit->papi_eventsets[PER_SOCKET_SET], info.values)) ) {
            dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                        exec_unit->th_id, PAPI_strerror(err));
            goto next_pins;
        }
        (void)dague_profiling_trace(exec_unit->eu_profile, exec_unit->pins_prof_papi_socket[exec_unit->begin_end],
                                    45, 0, (void *)&info);
        exec_unit->begin_end = (exec_unit->begin_end + 1) % 2;
    }

    next_pins:
    /* call previous callback, if any */
    if (NULL != exec_end_prev)
        (*exec_end_prev)(exec_unit, exec_context, data);

    (void)exec_context; (void)data;
}

const dague_pins_module_t dague_pins_papi_socket_module = {
    &dague_pins_papi_socket_component,
    {
        pins_init_papi_socket,
        pins_fini_papi_socket,
        NULL,
        NULL,
        pins_thread_init_papi_socket,
        pins_thread_fini_papi_socket,
    }
};
