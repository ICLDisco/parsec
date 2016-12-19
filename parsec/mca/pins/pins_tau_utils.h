/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PINS_TAU_UTILS_H
#define PINS_TAU_UTILS_H

#include "parsec.h"
#include "parsec/execution_unit.h"

void pins_tau_init(parsec_context_t * master_context);
void pins_tau_thread_init(parsec_execution_unit_t * exec_unit);

#endif
