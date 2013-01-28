#include <papi.h>
#include "cachemiss.h"
#include <stdio.h>
#include "execution_unit.h"
#include "src/pins/pins.h"

static int steal_events[NUM_STEAL_EVENTS] = {PAPI_L1_TCM, PAPI_L2_TCM};

#define NUM_EXEC_EVENTS 3
static int exec_events[NUM_EXEC_EVENTS] = {PAPI_L1_DCM, PAPI_L2_DCM, PAPI_TLB_TL};

int ExecEventSet = PAPI_NULL;
// extern int StealEventSet = PAPI_NULL;

void pins_init_cachemiss(dague_context_t * master_context) {
	(void)master_context;
    // PETER TODO put this in correct initialization place for PINS stuff
	// PINS_REGISTER(TASK_SELECT_BEGIN, start_papi_steal_count);

	PINS_REGISTER(EXEC_BEGIN, start_papi_exec_count);
	PINS_REGISTER(EXEC_FINI, stop_papi_exec_count);
}

void pins_thread_init_cachemiss(dague_execution_unit_t * exec_unit) {
	int rv;
	exec_unit->ExecEventSet = PAPI_NULL;
	printf("%p was PAPI_NULL\n", exec_unit);
	if (PAPI_create_eventset(&exec_unit->ExecEventSet) != PAPI_OK)
		printf("%p cachemiss.c, pins_thread_init_cachemiss: failed to create ExecEventSet\n", exec_unit);
	if ((rv = PAPI_add_events(exec_unit->ExecEventSet, exec_events, NUM_EXEC_EVENTS)) != PAPI_OK)
		printf("%p cachemiss.c, pins_thread_init_cachemiss: failed to add exec events to ExecEventSet. %d %s\n", exec_unit, rv, PAPI_strerror(rv));
    exec_unit->exec_cache_misses[0] = 0;
    exec_unit->exec_cache_misses[1] = 0;
    exec_unit->exec_tlb_misses = 0;
	

	exec_unit->StealEventSet = PAPI_NULL;
	if (PAPI_create_eventset(&exec_unit->StealEventSet) != PAPI_OK)
		printf("%p cachemiss.c, pins_thread_init_cachemiss: failed to create StealEventSet\n", exec_unit);
	if (PAPI_add_events(exec_unit->StealEventSet, steal_events, NUM_STEAL_EVENTS) != PAPI_OK)
		printf("%p cachemiss.c, pins_thread_init_cachemiss: failed to add steal events to StealEventSet\n", exec_unit);
}

void start_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	(void)exec_context;
	(void)data;
	int rv;
	if ((rv = PAPI_start(exec_unit->StealEventSet)) != PAPI_OK)
		printf("%p cachemiss.c, start_papi_steal_count: can't start steal event counters! %d %s\n", exec_unit, rv, PAPI_strerror(rv));
}

// PETER this is currently done by SCHED_STEAL/count_steals in pins/steals/steals.c
void stop_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	(void)exec_context;
	(void)data;
	long long int values[NUM_STEAL_EVENTS];
	int rv;
	printf("%p cachemiss.c, stop_papi_steal_count: this should not be running.\n", exec_unit);
	if ((rv = PAPI_stop(exec_unit->StealEventSet, values)) != PAPI_OK)
		printf("%p cachemiss.c, stop_papi_steal_count: can't stop steal event counters! %d %s\n", exec_unit, rv, PAPI_strerror(rv));
	else {
		// add counters
		exec_unit->select_counters[0] += values[0];
		exec_unit->select_counters[1] += values[1];
	}
}

void start_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	(void)exec_context;
	(void)data;
	int rv = PAPI_OK;
	if ((rv = PAPI_start(exec_unit->ExecEventSet)) != PAPI_OK)
		printf("%p cachemiss.c, start_papi_exec_count: can't start exec event counters! %d %s\n", exec_unit, rv, PAPI_strerror(rv));
}

void stop_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	(void)exec_context;
	(void)data;
	long long int values[NUM_EXEC_EVENTS];
	int rv = PAPI_OK;
	if ((rv = PAPI_stop(exec_unit->ExecEventSet, values)) != PAPI_OK)
		printf("%p cachemiss.c, stop_papi_exec_count: can't stop exec event counters! %d %s\n", exec_unit, rv, PAPI_strerror(rv));
	else {
		// add counters
		exec_unit->exec_cache_misses[0] += values[0];
		exec_unit->exec_cache_misses[1] += values[1];
		exec_unit->exec_tlb_misses += values[2];
	}
	/*
	if ((rv = PAPI_destroy_eventset(&exec_unit->ExecEventSet)) != PAPI_OK)
		printf("%p cachemiss.c, start_papi_exec_count: can't destroy exec event counters! %d %s\n", exec_unit, rv, PAPI_strerror(rv));
	else
		exec_unit->ExecEventSet = PAPI_NULL;
	 */
}

