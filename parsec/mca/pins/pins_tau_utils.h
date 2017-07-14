/*
 * Copyright (c) 2012-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PINS_TAU_UTILS_H
#define PINS_TAU_UTILS_H

#include "parsec.h"
#include "parsec/execution_stream.h"

void pins_tau_init(parsec_context_t * master_context);
void pins_tau_thread_init(parsec_execution_stream_t* es);

#endif
