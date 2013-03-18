#ifndef CACHEMISS_H
#define CACHEMISS_H
#include "dague_config.h"
#include "dague.h"
#ifdef HAVE_PAPI

#include <papi.h>

int pins_prof_exec_misses_start, pins_prof_exec_misses_stop;

#ifdef PINS_SHARED_L3_MISSES
#define DO_L3_MEASUREMENTS 1
#else
#define DO_L3_MEASUREMENTS 0
#endif

#define NUM_EXEC_EVENTS 4
static int exec_events[NUM_EXEC_EVENTS] = {PAPI_RES_STL, PAPI_L2_DCH, PAPI_L2_DCM, PAPI_L1_ICM};

typedef struct pins_cachemiss_info_s {
	int kernel_type;
	int th_id;
	int values_len;
	long long values[NUM_EXEC_EVENTS];
} pins_cachemiss_info_t;

void pins_init_cachemiss(dague_context_t * master_context);
void pins_handle_init_cachemiss(dague_handle_t * handle);
void pins_thread_init_cachemiss(dague_execution_unit_t * exec_unit);

void start_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void stop_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

#endif // HAVE_PAPI

#endif // CACHEMISS_H
