#include "steals.h"
#include "dague/pins/pins.h"
#include "debug.h"
#include "execution_unit.h"
#include <papi.h>

static unsigned int num_cores;
static unsigned int ** steals;
static int steal_events[NUM_STEAL_EVENTS] = {PAPI_L2_TCM, PAPI_L2_DCM};

#define THREAD_NUM(exec_unit) (exec_unit->virtual_process->vp_id *       \
                              exec_unit->virtual_process->dague_context->nb_vp + \
                              exec_unit->th_id )

/**
 NOTE: eu and task will be NULL under normal circumstances
 */
void pins_init_steals(dague_context_t * master) {
	unsigned int i;
	for (i = 0; i < master->nb_vp; i++)
		num_cores += master->virtual_processes[i]->nb_cores;

	// set up counters
	steals = (unsigned int**)calloc(sizeof(int *), num_cores);
	for (i = 0; i < num_cores; i++) {
		steals[i] = (unsigned int*)calloc(sizeof(int), num_cores + 2);
	}

	PINS_REGISTER(TASK_SELECT_BEGIN, start_papi_steal_count);
	PINS_REGISTER(TASK_SELECT_FINI, stop_papi_steal_count);
}

void pins_thread_init_steals(dague_execution_unit_t * exec_unit) {
	unsigned int p, t;
	dague_vp_t * vp = NULL;
	int rv;

	exec_unit->papi_eventsets[1] = PAPI_NULL;
	if (PAPI_create_eventset(&exec_unit->papi_eventsets[1]) != PAPI_OK)
		printf("steals.c, pins_thread_init_steals: failed to create StealEventSet\n");
	if ((rv = PAPI_add_events(exec_unit->papi_eventsets[1], steal_events, NUM_STEAL_EVENTS)) != PAPI_OK)
		printf("steals.c, pins_thread_init_steals: failed to add "
		       "steal events to StealEventSet. %d %s\n", rv, PAPI_strerror(rv));

	exec_unit->self_counters[0] = 0;
    exec_unit->self_counters[1] = 0;
    exec_unit->self = 1;
    exec_unit->steal_counters[0] = 0;
    exec_unit->steal_counters[1] = 0;
    exec_unit->steal = 1;
    exec_unit->other_counters[0] = 0;
    exec_unit->other_counters[1] = 0;
    exec_unit->other = 1;

    exec_unit->select_counters[0] = 0;
    exec_unit->select_counters[1] = 0;
}

void pins_fini_steals(dague_context_t * master_context) {
	(void)master_context;
	int p, q = 0;
	unsigned int i = 0;
	// unregister things
	PINS_UNREGISTER(TASK_SELECT_FINI);

	// print everything
	for (p = 0; p < num_cores; p++) {
		for (q = 0; q < num_cores + 2; q++) {
			printf("%6u ", steals[p][q]);
		}
		printf("\n");
	}

	for (i = 0; i < num_cores; i++) {
		free(steals[i]);
		steals[i] = NULL;
	}
	free(steals);
	steals = NULL;
}

// PETER this is all a hack
// PaRSEC would not compile with normal includes, because they were somehow circular?
#include "hbbuffer.h"
#include "list.h"
#include "dequeue.h"
typedef struct {
    dague_dequeue_t   *system_queue;
    dague_hbbuffer_t  *task_queue;
    int                nb_hierarch_queues;
    dague_hbbuffer_t **hierarch_queues;
} local_queues_scheduler_object_t;
#define LOCAL_QUEUES_OBJECT(eu_context) ((local_queues_scheduler_object_t*)(eu_context)->scheduler_object)
// END HACK

void start_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data) {
	(void)exec_context;
	(void)data;
	int rv;
	if ((rv = PAPI_start(exec_unit->papi_eventsets[1])) != PAPI_OK)
		printf("%p steals.c, start_papi_steal_count: can't start steal event counters! %d %s\n", exec_unit, rv, PAPI_strerror(rv));
}

void stop_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data) {
	// This section counts the steals (self, neighbor, system queue, starvation)
	unsigned int victim_core_num = (unsigned int)data;
	if (task != NULL) 
		steals[THREAD_NUM(exec_unit)][victim_core_num]++;
	else // starvation
		steals[THREAD_NUM(exec_unit)][victim_core_num + 1]++;

	// the rest of this code is a more comprehensive (but hacky!) counter of PAPI events during steals
	long long int values[NUM_STEAL_EVENTS];
	int rv = PAPI_OK;
	if ((rv = PAPI_stop(exec_unit->papi_eventsets[1], values)) != PAPI_OK)
		printf("steals.c, stop_papi_steal_count: can't stop steal event counters! %d %s\n", rv, PAPI_strerror(rv));
	else {
		unsigned int cur_core_num = LOCAL_QUEUES_OBJECT(exec_unit)->task_queue->assoc_core_num;
		// add counters
		if (cur_core_num == victim_core_num) {
			exec_unit->self_counters[0] += values[0];
			exec_unit->self_counters[1] += values[1];
			exec_unit->self++;
		}
		else if (victim_core_num == 48 || victim_core_num == 49) {
			exec_unit->other_counters[0] += values[0];
			exec_unit->other_counters[1] += values[1];
			exec_unit->other++;
		}
		else { // steal
			exec_unit->steal_counters[0] += values[0];
			exec_unit->steal_counters[1] += values[1];
			exec_unit->steal++;
		}
	}
}

/*
static unsigned int core_lookup(dague_execution_unit_t * eu) {
	unsigned int i = 0;
	for (; i < num_cores; i++) {
		if (eu == map[i])
			return i;
	}
	return 0;
}
 */

