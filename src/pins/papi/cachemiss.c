#include <papi.h>
#include "cachemiss.h"
#include <stdio.h>
#include "execution_unit.h"
#include "src/pins/pins.h"

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
#endif
}

void pins_handle_init_cachemiss(dague_handle_t * handle) {
	int rv;
	int num_cores = 0;
	unsigned int i;
	for (i = 0; i < handle->context->nb_vp; i++)
		num_cores += handle->context->virtual_processes[i]->nb_cores;
	
	printf("handle_cm: %p %p \n", handle, handle->pins_data[0]);
	for (i = 0; i < num_cores; i++) {
		printf("i is %u, EB is %d\n", i, EXEC_BEGIN);
		handle->pins_data[i][EXEC_BEGIN] = calloc(sizeof(pins_cachemiss_data_t), 1);
		DATA(handle, i)->ExecEventSet = PAPI_NULL;
		if (PAPI_create_eventset(&(DATA(handle, i)->ExecEventSet)) != PAPI_OK)
           printf("cachemiss.c, pins_handle_init_cachemiss: failed to create ExecEventSet\n");
       if ((rv = PAPI_add_events(DATA(handle, i)->ExecEventSet, exec_events, NUM_EXEC_EVENTS)) != PAPI_OK)
           printf("cachemiss.c, pins_handle_init_cachemiss: failed to add exec events to ExecEventSet. %d %s\n", rv, PAPI_strerror(rv));
       DATA(handle, i)->exec_cache_misses[0] = 0; // technically unnecessary b/c calloc
       DATA(handle, i)->exec_cache_misses[1] = 0;
       DATA(handle, i)->exec_tlb_misses = 0;

       /*
       DATA(handle, i)->StealEventSet = PAPI_NULL;
       if (PAPI_create_eventset(&DATA(handle, i)->StealEventSet) != PAPI_OK)
           printf("cachemiss.c, pins_handle_init_cachemiss: failed to create StealEventSet\n");
       if (PAPI_add_events(DATA(handle, i)->StealEventSet, steal_events, NUM_STEAL_EVENTS) != PAPI_OK)
           printf("cachemiss.c, pins_handle_init_cachemiss: failed to add steal events to StealEventSet\n");
        */
	}	

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
	printf("thread num %d\n", THREAD_NUM(exec_unit));
	if ((rv = PAPI_start(WDATA(exec_unit, exec_context)->ExecEventSet)) != PAPI_OK)
		printf("cachemiss.c, start_papi_exec_count: can't start exec event counters! %d %s\n", rv, PAPI_strerror(rv));
}

void stop_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	(void)exec_context;
	(void)data;
	long long int values[NUM_EXEC_EVENTS];
	int rv = PAPI_OK;
	if ((rv = PAPI_stop(WDATA(exec_unit, exec_context)->ExecEventSet, values)) != PAPI_OK)
		printf("cachemiss.c, stop_papi_exec_count: can't stop exec event counters! %d %s\n", rv, PAPI_strerror(rv));
	else {
		// add counters
		WDATA(exec_unit, exec_context)->exec_cache_misses[0] += values[0];
		WDATA(exec_unit, exec_context)->exec_cache_misses[1] += values[1];
		WDATA(exec_unit, exec_context)->exec_tlb_misses += values[2];
	}
}

