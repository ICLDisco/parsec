#include "cachemiss.h"
#include <stdio.h>
#include "execution_unit.h"

#define NUM_EVENTS 2
int events[NUM_EVENTS] = {PAPI_L1_TCM, PAPI_L2_TCM};

#define NUM_EXEC_EVENTS 2
int exec_events[NUM_EXEC_EVENTS] = {PAPI_L1_TCM, PAPI_L2_TCM, PAPI_TLB_TL};

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
		exec_unit->select_counters[0] += values[0];
		exec_unit->select_counters[1] += values[1];
	}
}

void start_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	int retval = PAPI_OK;
	//	PAPI_start_counters(exec_events, NUM_EXEC_EVENTS);
	//		printf("can't start exec event counters! %d %s\n", retval, PAPI_strerror(retval));
}

void stop_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	long long int values[NUM_EXEC_EVENTS] = {0, 0, 0};
	//	int retval = PAPI_OK;
	PAPI_stop_counters(values, NUM_EXEC_EVENTS);
	//	if ((retval = PAPI_stop_counters(values, NUM_EXEC_EVENTS)) != PAPI_OK)
	//		printf("can't stop exec event counters! %d %s\n", retval, PAPI_strerror(retval));
	//	else {
		// add counters
		exec_unit->exec_cache_misses[0] += values[0];
		exec_unit->exec_cache_misses[1] += values[1];
		exec_unit->exec_tlb_misses += values[2];
		//	}
}

