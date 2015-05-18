/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PINS_TAU_UTILS_H
#define PINS_TAU_UTILS_H

#include "dague.h"
#include "dague/execution_unit.h"

void pins_tau_init(dague_context_t * master_context);
void pins_tau_thread_init(dague_execution_unit_t * exec_unit);

#endif
