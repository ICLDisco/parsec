#ifndef SHARED_L3_MISSES_H
#define SHARED_L3_MISSES_H

#include "dague.h"

void pins_init_shared_L3_misses(dague_context_t * master_context);

void start_shared_L3_misses(dague_execution_unit_t * exec_unit, 
                            dague_execution_context_t * exec_context, void * data);
void stop_shared_L3_misses(dague_execution_unit_t * exec_unit, 
                           dague_execution_context_t * exec_context, void * data);


#endif



