#ifndef PINS_TAU_UTILS_H
#define PINS_TAU_UTILS_H

#include "dague.h"
#include "execution_unit.h"

void pins_tau_init(dague_context_t * master_context);
void pins_tau_thread_init(dague_execution_unit_t * exec_unit);

#endif
