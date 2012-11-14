#include "instru_steals.h"
#include "instrument.h"
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
void init_instru_steals(dague_execution_unit_t * eu, dague_execution_context_t * task, void * data) {
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
	printf("registered everything\n");
}

void fini_instru_steals(dague_execution_unit_t * eu, dague_execution_context_t * task, void * data) {
	(void) eu;
	(void) task;
	(void) data;
	int p, q = 0;
	unsigned int i = 0;
	// unregister things
	printf("entering fini\n");
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

void count_steal(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data) {
	unsigned int victim_core_num = (unsigned int)data;
	if (task != NULL) 
		steals[core_lookup(exec_unit)][victim_core_num]++;
	else // starvation
		steals[core_lookup(exec_unit)][victim_core_num + 1]++;
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

