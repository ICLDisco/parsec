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

static int pins_prof_papi_socket[2];

static char* mca_param_string;

static void pins_init_papi_socket(dague_context_t * master_context)
{
    pins_papi_init(master_context);

    dague_mca_param_reg_string_name("pins", "socket_event",
				    "PAPI event to be saved.\n",
				    false, false,
				    "", &mca_param_string);

    /* Handcrafted event to be added to the profiling */
    /* The first argument has to be "PINS_EXEC" for now. */
    dague_profiling_add_dictionary_keyword("PINS_SOCKET", "fill:#00AAFF",
					   sizeof(papi_socket_info_t), "value1{int64_t}:value2{int64_t}",
					   &pins_prof_papi_socket[0],
					   &pins_prof_papi_socket[1]);

    /* prepare link to the previously registered PINS module */
    exec_end_prev = PINS_REGISTER(EXEC_END, stop_papi_socket);
}

static void pins_fini_papi_socket(dague_context_t * master_context)
{
  (void) master_context;

  PINS_REGISTER(EXEC_END, exec_end_prev);
}

static void pins_thread_init_papi_socket(dague_execution_unit_t * exec_unit)
{
    char* mca_param_name;
    char* token;
    char* temp;
    int err, i;
    bool socket, core, started = false;

    exec_unit->num_socket_counters = 0;
    exec_unit->num_socket_tasks = 0;
    exec_unit->begin_end = 0;
    exec_unit->pins_papi_socket_event_name = (char**)calloc(NUM_SOCKET_EVENTS, sizeof(char*));
    exec_unit->pins_papi_socket_native_event = (int*)calloc(NUM_SOCKET_EVENTS, sizeof(int));
    exec_unit->papi_eventsets[PER_SOCKET_SET] = PAPI_NULL;

    for(i = 0; i < NUM_SOCKET_EVENTS; i++) {
        exec_unit->pins_papi_socket_event_name[i] = NULL;
        exec_unit->pins_papi_socket_native_event[i] = PAPI_NULL;
    }

    mca_param_name = strdup(mca_param_string);
    token = strtok(mca_param_name, ":");

    if(token == NULL) {
        dague_output(0, "No PAPI events have been specified.  None will be recorded.\n");
        exec_unit->papi_eventsets[PER_SOCKET_SET] = PAPI_NULL;
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

        token = strtok(NULL, ":");

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

        token = strtok(NULL, ",");

        if(socket && core) {
            if(exec_unit->num_socket_counters == NUM_SOCKET_EVENTS) {
                dague_output(0, "pins_thread_init_papi_socket: thread %d couldn't add event '%s' because only %d events are allowed.\n",
                             exec_unit->th_id, token, NUM_SOCKET_EVENTS);
                break;
	    }

            if(!started) {
                pins_papi_thread_init(exec_unit);
                exec_unit->papi_eventsets[PER_SOCKET_SET] = PAPI_NULL;

                /* Create an empty eventset */
                if( PAPI_OK != (err = PAPI_create_eventset(&exec_unit->papi_eventsets[PER_SOCKET_SET])) ) {
                    dague_output(0, "pins_thread_init_papi_socket: thread %d couldn't create the PAPI event set; ERROR: %s\n",
                                 exec_unit->th_id, PAPI_strerror(err));
                    return;
		}
                started = true;
	    }

            /* Convert event name to code */
            if(PAPI_OK == PAPI_event_name_to_code(token, &exec_unit->pins_papi_socket_native_event[exec_unit->num_socket_counters]) ) {
                exec_unit->pins_papi_socket_event_name[exec_unit->num_socket_counters] = strdup(token);
	    }

            if(PAPI_NULL == exec_unit->pins_papi_socket_native_event[exec_unit->num_socket_counters]) {
                dague_output(0, "No event derived from %s is supported on this system (use papi_native_avail for a complete list)\n",
                             token);
                return;
	    }

            /* Add events to the eventset */
            if( PAPI_OK != (err = PAPI_add_event(exec_unit->papi_eventsets[PER_SOCKET_SET],
                                                 exec_unit->pins_papi_socket_native_event[exec_unit->num_socket_counters])) ) {
                dague_output(0, "pins_thread_init_papi_socket: failed to add event %s; ERROR: %s\n",
                             token, PAPI_strerror(err));
                return;
	    }

            exec_unit->num_socket_counters++;
	}

        token = strtok(NULL, ":");
    }

    free(mca_param_name);
    free(token);

    if(exec_unit->num_socket_counters > 0) {
        papi_socket_info_t info;

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

        (void)dague_profiling_trace(exec_unit->eu_profile,
                                    pins_prof_papi_socket[exec_unit->begin_end], 45, 0, (void *)&info);
        exec_unit->begin_end = (exec_unit->begin_end + 1) % 2;
    }
}

static void pins_thread_fini_papi_socket(dague_execution_unit_t * exec_unit)
{
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
            (void)dague_profiling_trace(exec_unit->eu_profile,
                                        pins_prof_papi_socket[0], 45, 0, (void *)&info);
            (void)dague_profiling_trace(exec_unit->eu_profile,
                                        pins_prof_papi_socket[1], 45, 0, (void *)&info);
	} else
            (void)dague_profiling_trace(exec_unit->eu_profile,
                                        pins_prof_papi_socket[1], 45, 0, (void *)&info);
    }
    /* the counting should be stopped by now */
    for(i = 0; i < NUM_SOCKET_EVENTS; i++) {
        if( PAPI_OK != (err = PAPI_remove_event(exec_unit->papi_eventsets[PER_SOCKET_SET],
                                                exec_unit->pins_papi_socket_native_event[i])) ) {
            dague_output(0, "pins_thread_fini_papi_socket: failed to remove event %s; ERROR: %s\n",
                         exec_unit->pins_papi_socket_event_name[i], PAPI_strerror(err));
	}
    }

    for(i = 0; i < NUM_SOCKET_EVENTS; i++) {
        free(exec_unit->pins_papi_socket_event_name[i]);
    }

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
			     void * data)
{
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
        printf("dumping profiling %ld %ld\n", info.values[0], info.values[1]);
        (void)dague_profiling_trace(exec_unit->eu_profile,
                                    pins_prof_papi_socket[exec_unit->begin_end], 45, 0, (void *)&info);
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
