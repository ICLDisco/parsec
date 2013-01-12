#include "steals.h"
#include "pins.h"
#include "debug.h"
#include "execution_unit.h"

static unsigned int num_cores;
static unsigned int ** steals;
/* array map should preferably be replaced with a hash map */
static dague_execution_unit_t ** map;

static unsigned int core_lookup(dague_execution_unit_t * eu);

/**
 NOTE: eu and task will be NULL under normal circumstances
 */
void pins_init_steals(dague_execution_unit_t * eu, dague_execution_context_t * task, void * data) {
	(void) eu;
	(void) task;
	unsigned int i, p, t = 0;
	dague_context_t * master = (dague_context_t *)data;
	dague_vp_t * vp = NULL;
	num_cores = master->nb_vp * master->virtual_processes[0]->nb_cores;

	// set up map
	map = (dague_execution_unit_t**)calloc(sizeof(dague_execution_unit_t *), num_cores);
	for (p = 0; p < master->nb_vp; p++) {
		vp = master->virtual_processes[p];
		for (t = 0; t < vp->nb_cores; t++) {
			map[p * vp->nb_cores + t] = vp->execution_units[t];
		}
	}

	// set up counters
	steals = (unsigned int**)calloc(sizeof(int *), num_cores);
	for (i = 0; i < num_cores; i++) {
		steals[i] = (unsigned int*)calloc(sizeof(int), num_cores + 2);
	}

	register_instrument_callback(SCHED_FINI, fini_instru_steals);
	register_instrument_callback(SCHED_STEAL, count_steal);
}

void pins_fini_steals(dague_execution_unit_t * eu, dague_execution_context_t * task, void * data) {
	(void) eu;
	(void) task;
	(void) data;
	int p, q = 0;
	unsigned int i = 0;
	// unregister things
	unregister_instrument_callback(SCHED_STEAL);

	// print everything
	for (p = 0; p < num_cores; p++) {
		for (q = 0; q < num_cores + 2; q++) {
			printf("%4u ", steals[p][q]);
		}
		printf("\n");
	}

	// free everything
	free(map);
	map = NULL;

	for (i = 0; i < num_cores; i++) {
		free(steals[i]);
		steals[i] = NULL;
	}
	free(steals);
	steals = NULL;
}

// PETER this is all a hack
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
#define NUM_EVENTS 2
#include <papi.h>
void count_steal(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data) {
	unsigned int victim_core_num = (unsigned int)data;
	if (task != NULL) 
		steals[core_lookup(exec_unit)][victim_core_num]++;
	else // starvation
		steals[core_lookup(exec_unit)][victim_core_num + 1]++;

	long long int values[NUM_EVENTS] = {0, 0};
	int retval = PAPI_OK;
	if ((retval = PAPI_stop_counters(values, NUM_EVENTS)) != PAPI_OK)
		printf("can't stop event counters! %d %s\n", retval, PAPI_strerror(retval));
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

static unsigned int core_lookup(dague_execution_unit_t * eu) {
	unsigned int i = 0;
	for (; i < num_cores; i++) {
		if (eu == map[i])
			return i;
	}
	DEBUG(("Core lookup in scheduling instrumentation failed. Returning 0.\n"));
	return 0;
}

