#ifndef CACHEMISS_H
#define CACHEMISS_H
#include "dague.h"

#define NUM_STEAL_EVENTS 2
int StealEventSet;

void pins_init_cachemiss(dague_context_t * master_context);
void pins_handle_init_cachemiss(dague_handle_t * handle);
void pins_thread_init_cachemiss(dague_execution_unit_t * exec_unit);

void start_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void stop_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void start_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void stop_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

#endif
