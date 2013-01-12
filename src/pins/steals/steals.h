#ifndef STEALS_H
#define STEALS_H
#include "dague.h"

void pins_init_steals(dague_execution_unit_t * eu, dague_execution_context_t * task, void * data);
void pins_fini_steals(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data);
void count_steal(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

#endif
