#ifndef STEALS_H
#define STEALS_H
#include "dague.h"

void pins_init_steals(dague_context_t * master_context);
void pins_handle_init_steals(dague_handle_t * handle);
void pins_fini_steals(dague_context_t * master_context);

void count_steal(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

#endif
