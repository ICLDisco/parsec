#include "cachemiss.h"
#include <stdio.h>
#include "execution_unit.h"

#define NUM_EVENTS 2
int events[NUM_EVENTS] = {PAPI_L1_TCM, PAPI_L2_TCM};

void start_papi_cache_miss_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	int retval = PAPI_OK;
	if ((retval = PAPI_start_counters(events, NUM_EVENTS)) != PAPI_OK)
		printf("can't start event counters! %d %s\n", retval, PAPI_strerror(retval));
}

void stop_papi_cache_miss_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	long long int values[NUM_EVENTS] = {0, 0};
	int retval = PAPI_OK;
	if ((retval = PAPI_stop_counters(values, NUM_EVENTS)) != PAPI_OK)
		printf("can't stop event counters! %d %s\n", retval, PAPI_strerror(retval));
	else {
		// add counters
		//	exec_unit->counters[0] += values[0];
		// exec_unit->counters[1] += values[1];
	}
}


// NOTE TO SELF: what is this return value '1', and where is it coming from!?
