#ifndef INSTRU_STEALS_H
#define INSTRU_STEALS_H
#include "dague.h"

void init_instru_steals(dague_execution_unit_t * eu, dague_execution_context_t * task, void * data);
void fini_instru_steals(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data);
void count_steal(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

#endif
