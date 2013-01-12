#ifndef CACHEMISS_H
#define CACHEMISS_H
#include "dague.h"
#include <papi.h>

void start_papi_cache_miss_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void stop_papi_cache_miss_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

#endif
