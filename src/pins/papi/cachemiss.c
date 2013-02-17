#include "dague_config.h"
#ifdef HAVE_PAPI
#include <papi.h>
#include "cachemiss.h"
#include <stdio.h>
#include "execution_unit.h"
#include "profiling.h"
#include "src/pins/pins.h"

int pins_prof_exec_misses_start, pins_prof_exec_misses_stop;

static int steal_events[NUM_STEAL_EVENTS] = {PAPI_L1_TCM, PAPI_L2_TCM};

#define NUM_EXEC_EVENTS 3
static int exec_events[NUM_EXEC_EVENTS] = {PAPI_L1_DCM, PAPI_L2_DCM, PAPI_TLB_TL};

typedef struct pins_cachemiss_data_s {
	int ExecEventSet;
	exec_cache_misses[2];
	exec_tlb_misses;
} pins_cachemiss_data_t;

#define THREAD_NUM(exec_unit) (exec_unit->virtual_process->vp_id *       \
                              exec_unit->virtual_process->dague_context->nb_vp + \
                              exec_unit->th_id )

#define DATA(handle, i) ((pins_cachemiss_data_t*)handle->pins_data[i][EXEC_BEGIN])
#define WDATA(exec_unit, exec_task) ((pins_cachemiss_data_t*)exec_task->dague_handle->pins_data[THREAD_NUM(exec_unit)][EXEC_BEGIN])

void pins_init_cachemiss(dague_context_t * master_context) {
	(void)master_context;
    // PETER TODO put this in correct initialization place for PINS stuff
#ifdef PINS_SCHED_STEALS
	PINS_REGISTER(TASK_SELECT_BEGIN, start_papi_steal_count);
#endif

#ifdef PINS_EXEC_MISSES
	PINS_REGISTER(EXEC_BEGIN, start_papi_exec_count);
	PINS_REGISTER(EXEC_FINI, stop_papi_exec_count);
	// PETER TODO add requirement for DAGUE_PROF_TRACE
	dague_profiling_add_dictionary_keyword("EXEC_MISSES", "fill:#00FF00",
	                                       sizeof(long long), NULL,
	                                       &pins_prof_exec_misses_start, &pins_prof_exec_misses_stop);
#endif
}

void pins_thread_init_cachemiss(dague_execution_unit_t * exec_unit) {
	int rv = 0;
	exec_unit->papi_eventsets[0] = PAPI_NULL;
	exec_unit->papi_eventsets[1] = PAPI_NULL;
	if (PAPI_create_eventset(&exec_unit->papi_eventsets[0]) != PAPI_OK)
		printf("cachemiss.c, pins_thread_init_cachemiss: failed to create ExecEventSet\n");
	if ((rv = PAPI_add_events(exec_unit->papi_eventsets[0], exec_events, NUM_EXEC_EVENTS)) != PAPI_OK)
		printf("cachemiss.c, pins_thread_init_cachemiss: failed to add exec events to ExecEventSet. %d %s\n", rv, PAPI_strerror(rv));
	if (PAPI_create_eventset(&exec_unit->papi_eventsets[1]) != PAPI_OK)
		printf("cachemiss.c, pins_thread_init_cachemiss: failed to create StealEventSet\n");
}

void pins_handle_init_cachemiss(dague_handle_t * handle) {
	/*
	int rv;
	int num_cores = 0;
	unsigned int i;
	for (i = 0; i < handle->context->nb_vp; i++)
		num_cores += handle->context->virtual_processes[i]->nb_cores;
	
	for (i = 0; i < num_cores; i++) {
		handle->pins_data[i][EXEC_BEGIN] = calloc(sizeof(pins_cachemiss_data_t), 1);
		DATA(handle, i)->ExecEventSet = PAPI_NULL;
		if (PAPI_create_eventset(&(DATA(handle, i)->ExecEventSet)) != PAPI_OK)
			DEBUG(("cachemiss.c, pins_handle_init_cachemiss: failed to create ExecEventSet\n"));
       if ((rv = PAPI_add_events(DATA(handle, i)->ExecEventSet, exec_events, NUM_EXEC_EVENTS)) != PAPI_OK)
	       DEBUG(("cachemiss.c, pins_handle_init_cachemiss: failed to add exec events to ExecEventSet. %d %s\n", rv, PAPI_strerror(rv)));
       DATA(handle, i)->exec_cache_misses[0] = 0; // technically unnecessary b/c calloc
       DATA(handle, i)->exec_cache_misses[1] = 0;
       DATA(handle, i)->exec_tlb_misses = 0;

       DATA(handle, i)->StealEventSet = PAPI_NULL;
       if (PAPI_create_eventset(&DATA(handle, i)->StealEventSet) != PAPI_OK)
           printf("cachemiss.c, pins_handle_init_cachemiss: failed to create StealEventSet\n");
       if (PAPI_add_events(DATA(handle, i)->StealEventSet, steal_events, NUM_STEAL_EVENTS) != PAPI_OK)
           printf("cachemiss.c, pins_handle_init_cachemiss: failed to add steal events to StealEventSet\n");
        
	}	
	 */
}

/*
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
 */

void start_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	(void)exec_context;
	(void)data;
	int rv = PAPI_OK;
	if ((rv = PAPI_start(exec_unit->papi_eventsets[0])) != PAPI_OK)
	    printf("cachemiss.c, start_papi_exec_count: can't start exec event counters! %d %s\n", rv, PAPI_strerror(rv));
	else {
		dague_profiling_trace(exec_unit->eu_profile, pins_prof_exec_misses_start, 0, -1, NULL);
	}
		    
}

void stop_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	(void)exec_context;
	(void)data;
	long long int values[NUM_EXEC_EVENTS];
	int rv = PAPI_OK;
	if ((rv = PAPI_stop(exec_unit->papi_eventsets[0], values)) != PAPI_OK)
		printf("cachemiss.c, stop_papi_exec_count: can't stop exec event counters! %d %s\n", rv, PAPI_strerror(rv));
	else {
		dague_profiling_trace(exec_unit->eu_profile, pins_prof_exec_misses_stop, 0, -1, &values[0]);
	}
}

#endif
