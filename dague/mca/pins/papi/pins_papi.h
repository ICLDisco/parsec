/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef MCA_PINS_PAPI_H
#define MCA_PINS_PAPI_H

#include "dague.h"
#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_papi_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_papi_module;
/* static accessor */
mca_base_component_t * pins_papi_static_component(void);

END_C_DECLS

#endif



