#include "steals.h"
#include "dague/pins/pins.h"
#include "debug.h"
#include "execution_unit.h"
#include <papi.h>

static unsigned int num_cores;
static unsigned int ** steals;
static int pins_prof_task_select_begin, pins_prof_task_select_end;
static int steal_events[NUM_TASK_SELECT_EVENTS] = {PAPI_L2_TCM, PAPI_L2_DCM};

#define THREAD_NUM(exec_unit) (exec_unit->virtual_process->vp_id *       \
                              exec_unit->virtual_process->dague_context->nb_vp + \
                              exec_unit->th_id )

/**
 NOTE: eu and task will be NULL under normal circumstances
 */
void pins_init_steals(dague_context_t * master) {
	PINS_REGISTER(TASK_SELECT_BEGIN, start_papi_steal_count);
	PINS_REGISTER(TASK_SELECT_END, stop_papi_steal_count);
}

void pins_thread_init_steals(dague_execution_unit_t * exec_unit) {
	unsigned int p, t;
	dague_vp_t * vp = NULL;
	int rv;

	exec_unit->papi_eventsets[1] = PAPI_NULL;
	if (PAPI_create_eventset(&exec_unit->papi_eventsets[TASK_SELECT_SET]) != PAPI_OK)
		printf("steals.c, pins_thread_init_steals: failed to create StealEventSet\n");
	if ((rv = PAPI_add_events(exec_unit->papi_eventsets[TASK_SELECT_SET], steal_events, NUM_TASK_SELECT_EVENTS)) != PAPI_OK)
		printf("steals.c, pins_thread_init_steals: failed to add "
		       "steal events to StealEventSet. %d %s\n", rv, PAPI_strerror(rv));
	dague_profiling_add_dictionary_keyword("PINS_TASK_SELECT", "fill:#0000FF",
	                                       sizeof(pins_task_select_info_t), NULL,
	                                       &pins_prof_task_select_begin, &pins_prof_task_select_end);
}

void start_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	(void)exec_context;
	(void)data;
	int rv;
	if ((rv = PAPI_start(exec_unit->papi_eventsets[TASK_SELECT_SET])) != PAPI_OK)
		printf("%p steals.c, start_papi_steal_count: can't start steal event counters! %d %s\n", exec_unit, rv, PAPI_strerror(rv));
	else
		dague_profiling_trace(exec_unit->eu_profile, pins_prof_task_select_begin, 
		                      45,
		                      -2, NULL);
}

void stop_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	// This section counts the steals (self, neighbor, system queue, starvation)
	unsigned long long victim_core_num = (unsigned long long)data;
	unsigned int num_threads = (exec_unit->virtual_process->dague_context->nb_vp 
	                            * exec_unit->virtual_process->nb_cores);
	pins_task_select_info_t info;
	if (exec_context) 
		info.kernel_type = exec_context->function->function_id;
	else
		info.kernel_type = 0;
	info.vp_id = exec_unit->virtual_process->vp_id;
	info.th_id = exec_unit->th_id;
	info.victim_vp_id = -1; // currently unavailable from scheduler queue object
	if (victim_core_num >= num_threads)
		info.victim_vp_id = SYSTEM_QUEUE_VP;
	info.victim_th_id = (int)victim_core_num; // but this number includes the vp id multiplier
	info.exec_context = (unsigned long long int)exec_context; // if NULL, this was starvation

	// now count the PAPI events, if available
	long long int values[NUM_TASK_SELECT_EVENTS];
	int rv = PAPI_OK;
	if ((rv = PAPI_stop(exec_unit->papi_eventsets[TASK_SELECT_SET], values)) != PAPI_OK) {
		printf("steals.c, stop_papi_steal_count: can't stop steal event counters! %d %s\n", rv, PAPI_strerror(rv));
		for (int i = 0; i < NUM_TASK_SELECT_EVENTS; i++)
			info.values[i] = -1;
		info.values_len = 0;
	}
	else {
		for (int i = 0; i < NUM_TASK_SELECT_EVENTS; i++)
			info.values[i] = values[i];
		info.values_len = NUM_TASK_SELECT_EVENTS; /* see cachemiss.c for why this is included */
	}
	
	dague_profiling_trace(exec_unit->eu_profile, pins_prof_task_select_end, 
	                      45,
	                      -2, 
	                      (void *)&info);
}

