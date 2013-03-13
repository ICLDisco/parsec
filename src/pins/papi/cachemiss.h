#ifndef CACHEMISS_H
#define CACHEMISS_H
#include "dague.h"
#include <papi.h>

int pins_prof_exec_misses_start, pins_prof_exec_misses_stop;

#define NUM_STEAL_EVENTS 2
#define NUM_EXEC_EVENTS 4
static int exec_events[NUM_EXEC_EVENTS] = {PAPI_RES_STL, PAPI_L2_DCH, PAPI_L2_DCM, PAPI_L1_ICM};

typedef struct pins_cachemiss_info_s {
	int kernel_type;
	int th_id;
	int values_len;
	int values[NUM_EXEC_EVENTS];
} pins_cachemiss_info_t;

void pins_init_cachemiss(dague_context_t * master_context);
void pins_handle_init_cachemiss(dague_handle_t * handle);
void pins_thread_init_cachemiss(dague_execution_unit_t * exec_unit);

void start_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void stop_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void start_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void stop_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

#endif
