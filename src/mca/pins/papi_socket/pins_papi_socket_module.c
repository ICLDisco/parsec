#include "pins_papi_socket.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include <stdio.h>
#include <papi.h>
#include "execution_unit.h"

static void pins_init_papi_socket(dague_context_t * master_context);
static void pins_fini_papi_socket(dague_context_t * master_context);

const dague_pins_module_t dague_pins_papi_socket_module = {
    &dague_pins_papi_socket_component,
    {
	    pins_init_papi_socket,
	    pins_fini_papi_socket,
	    NULL,
	    NULL,
	    NULL,
	    NULL
    }
};

static void start_papi_socket(dague_execution_unit_t * exec_unit, 
                              dague_execution_context_t * exec_context, 
                              void * data);
static void stop_papi_socket(dague_execution_unit_t * exec_unit, 
                             dague_execution_context_t * exec_context, 
                             void * data);

static parsec_pins_callback * thread_init_prev; // courtesy calls to previously-registered cbs
static parsec_pins_callback * thread_fini_prev;

static char* socket_events [NUM_SOCKET_EVENTS] = {"L3_CACHE_MISSES:READ_BLOCK_EXCLUSIVE",
												  "L3_CACHE_MISSES:READ_BLOCK_SHARED",
												  "L3_CACHE_MISSES:READ_BLOCK_MODIFY"
												   };

static int successful_events = 0;

static void pins_init_papi_socket(dague_context_t * master_context) {
	thread_init_prev = PINS_REGISTER(THREAD_INIT, start_papi_socket);
	thread_fini_prev = PINS_REGISTER(THREAD_FINI, stop_papi_socket);
}

static void pins_fini_papi_socket(dague_context_t * master_context) {
	PINS_REGISTER(THREAD_INIT, thread_init_prev);
	PINS_REGISTER(THREAD_FINI, thread_fini_prev);
}

static void start_papi_socket(dague_execution_unit_t * exec_unit, 
                              dague_execution_context_t * exec_context, 
                              void * data) {
	unsigned int native;
	if (exec_unit->th_id % CORES_PER_SOCKET == WHICH_CORE_IN_SOCKET) {
		exec_unit->papi_eventsets[PER_SOCKET_SET] = PAPI_NULL;
		if (PAPI_create_eventset(&exec_unit->papi_eventsets[PER_SOCKET_SET]) != PAPI_OK)
			printf("couldn't create the PAPI event set for thread %d to measure L3 misses\n", exec_unit->th_id);
		else {
			int i = 0;
			int successful = 0;
			for (i = 0; i < NUM_SOCKET_EVENTS; i++) {
				if (PAPI_event_name_to_code(socket_events[i], &native) != PAPI_OK)
					printf("papi_socket couldn't find event %s.\n", socket_events[i]);
				else if (PAPI_add_event(exec_unit->papi_eventsets[PER_SOCKET_SET], native) != PAPI_OK)
					printf("papi_socket couldn't add event %s.\n", socket_events[i]);
				else
					successful += 1;
			}
			successful_events = successful;

			if (PAPI_start(exec_unit->papi_eventsets[PER_SOCKET_SET]) != PAPI_OK)
				printf("couldn't start PAPI event set for thread %d to measure L3 misses\n", exec_unit->th_id);
			else {
				//				printf("# started PAPI event set %d for thread %d (%p) to measure L3 misses\n", exec_unit->papi_eventsets[0], exec_unit->th_id, exec_unit);
				//				dague_profiling_trace(exec_unit->eu_profile, pins_prof_exec_misses_start, 45, 3, NULL);
				// this call to profiling trace is useless since we currently can't send the END event before the profile is dumped
			}
		}
	}
	// call previous callback, if any
	if (NULL != thread_init_prev) {
		(*thread_init_prev)(exec_unit, exec_context, data);
	}
}

static void stop_papi_socket(dague_execution_unit_t * exec_unit, 
                             dague_execution_context_t * exec_context, 
                             void * data) {
	(void)exec_context;
	(void)data;
	if (exec_unit->th_id % CORES_PER_SOCKET == WHICH_CORE_IN_SOCKET) {
		long long int values[NUM_SOCKET_EVENTS];
		int rv = PAPI_OK;
		if ((rv = PAPI_stop(exec_unit->papi_eventsets[PER_SOCKET_SET], values)) != PAPI_OK) {
			printf("couldn't stop PAPI event set %d for thread %d (%p) to measure L3 misses; ERROR:  %s\n", exec_unit->papi_eventsets[PER_SOCKET_SET], exec_unit->th_id, exec_unit, PAPI_strerror(rv));
		}
		else {
			char * buf = calloc(sizeof(char), (successful_events + 1) * 20);
			int inc = 0;
			char * buf_ptr = buf;
			long long int total = 0;
			for (int i = 0; i < successful_events; i++) {
				total += values[i];
				inc = snprintf(buf_ptr, 17, "%15lld ", values[i]);
				buf_ptr = (char *)buf_ptr + inc;
			}
			snprintf(buf_ptr, 21, "tot: %15lld", total);
			printf("%s\n", buf);
			free(buf);
			buf = NULL;
		}
	}
	// call previous callback, if any
	if (NULL != thread_fini_prev) {
		(*thread_fini_prev)(exec_unit, exec_context, data);
	}
}
