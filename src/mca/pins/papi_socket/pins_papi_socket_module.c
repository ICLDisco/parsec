#include "pins_papi_socket.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include <stdio.h>
#include <papi.h>
#include "execution_unit.h"

static void start_papi_socket(dague_execution_unit_t * exec_unit, 
                              dague_execution_context_t * exec_context, 
                              void * data);
static void stop_papi_socket(dague_execution_unit_t * exec_unit, 
                             dague_execution_context_t * exec_context, 
                             void * data);

static parsec_pins_callback * thread_init_prev; // courtesy calls to previously-registered cbs
static parsec_pins_callback * thread_fini_prev;
// TODO PETER finish simplifying this code, then create component.c for all modules
static char* select_events [NUM_SOCKET_EVENTS] = {};

void pins_init_papi_socket(dague_context_t * master_context) {
	thread_init_prev = PINS_REGISTER(THREAD_INIT, start_papi_socket);
	thread_fini_prev = PINS_REGISTER(THREAD_FINI, stop_papi_socket);
}

static void start_papi_socket(dague_execution_unit_t * exec_unit, 
                              dague_execution_context_t * exec_context, 
                              void * data) {
	(void)exec_context;
	(void)data;
	unsigned int native;
	if (exec_unit->th_id % CORES_PER_SOCKET == WHICH_CORE_IN_SOCKET
	    && DO_SOCKET_MEASUREMENTS) {
		exec_unit->papi_eventsets[PER_SOCKET_SET] = PAPI_NULL;
		if (PAPI_create_eventset(&exec_unit->papi_eventsets[PER_SOCKET_SET]) != PAPI_OK)
			printf("couldn't create the PAPI event set for thread %d to measure L3 misses\n", exec_unit->th_id);
		else {
			if (PAPI_event_name_to_code("L3_CACHE_MISSES:ANY_READ", &native) != PAPI_OK)
				printf("couldn't find L3_CACHE_MISSES:READ_BLOCK_EXCLUSIVE.\n");
			if (PAPI_add_event(exec_unit->papi_eventsets[PER_SOCKET_SET], native) != PAPI_OK)
				printf("couldn't add L3_CACHE_MISSES:READ_BLOCK_EXCLUSIVE.\n");

			if (PAPI_event_name_to_code("L3_CACHE_MISSES:READ_BLOCK_EXCLUSIVE", &native) != PAPI_OK)
				printf("couldn't find L3_CACHE_MISSES:READ_BLOCK_SHARED.\n");
			if (PAPI_add_event(exec_unit->papi_eventsets[PER_SOCKET_SET], native) != PAPI_OK)
				printf("couldn't add L3_CACHE_MISSES:READ_BLOCK_SHARED.\n");

			if (PAPI_event_name_to_code("L3_CACHE_MISSES:READ_BLOCK_MODIFY", &native) != PAPI_OK)
				printf("couldn't find DATA_CACHE_MISSES.\n");
			if (PAPI_add_event(exec_unit->papi_eventsets[PER_SOCKET_SET], native) != PAPI_OK)
				printf("couldn't add DATA_CACHE_MISSES.\n");

			if (PAPI_event_name_to_code("L3_CACHE_MISSES:READ_BLOCK_SHARED", &native) != PAPI_OK)
				printf("couldn't find PERF_COUNT_HW_INSTRUCTIONS.\n");
			if (PAPI_add_event(exec_unit->papi_eventsets[PER_SOCKET_SET], native) != PAPI_OK)
				printf("couldn't add PERF_COUNT_HW_INSTRUCTIONS.\n");

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
	if (exec_unit->th_id % CORES_PER_SOCKET == WHICH_CORE_IN_SOCKET 
	    && DO_SOCKET_MEASUREMENTS) {
		long long int values[NUM_SOCKET_EVENTS];
		int rv = PAPI_OK;
		if ((rv = PAPI_stop(exec_unit->papi_eventsets[PER_SOCKET_SET], values)) != PAPI_OK) {
			printf("couldn't stop PAPI event set %d for thread %d (%p) to measure L3 misses; ERROR:  %s\n", exec_unit->papi_eventsets[PER_SOCKET_SET], exec_unit->th_id, exec_unit, PAPI_strerror(rv));
		}
		else {
			char * buf = calloc(sizeof(char), NUM_SOCKET_EVENTS * 20);
			int inc = 0;
			for (int i = 0; i < NUM_SOCKET_EVENTS; i++) {
				inc = snprintf(buf, 17, "%15lld ", values[i]);
				buf = (char *)buf + inc;
			}
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
