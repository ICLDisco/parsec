#ifndef STEALS_H
#define STEALS_H
#include "dague.h"
#include <papi.h>

#define NUM_STEAL_EVENTS 2
int steal_events[NUM_STEAL_EVENTS] = {PAPI_L1_TCM, PAPI_L2_TCM};

void pins_init_steals(dague_context_t * master_context);
void pins_handle_init_steals(dague_handle_t * handle);
void pins_fini_steals(dague_context_t * master_context);

void start_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void stop_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

#endif
