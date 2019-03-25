/*
 * Copyright (c) 2018      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef PINS_ALPERF_H
#define PINS_ALPERF_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/pins/pins.h"
#include "parsec.h"

BEGIN_C_DECLS

/**
 * This module inspects the profiling dictionary on every event, looking for the corresponding task/event.
 * when found as a property to export to the user, the module will evaluate a function associated to the
 * property and write it down to a shared memory area. 
 */

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_pins_base_component_t parsec_pins_alperf_component;
PARSEC_DECLSPEC extern const parsec_pins_module_t parsec_pins_alperf_module;
/* static accessor */
mca_base_component_t * pins_alperf_static_component(void);


END_C_DECLS

#endif // PINS_ALPERF_H
