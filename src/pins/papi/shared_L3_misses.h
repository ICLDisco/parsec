#ifndef SHARED_L3_MISSES_H
#define SHARED_L3_MISSES_H

#define WHICH_CORE_FOR_L3 1 //mostly, just don't choose 0; it interferes with PaRSEC's thread handling
#define CORES_PER_SOCKET 6 // for ig.icl.utk.edu, an Istanbul Opteron, anyway

#include "dague.h"

void pins_init_shared_L3_misses(dague_context_t * master_context);

void start_shared_L3_misses(dague_execution_unit_t * exec_unit, 
                            dague_execution_context_t * exec_context, void * data);
void stop_shared_L3_misses(dague_execution_unit_t * exec_unit, 
                           dague_execution_context_t * exec_context, void * data);


#endif



