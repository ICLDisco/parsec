/*
 * Copyright (c) 2012-2015 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef MCA_PINS_PAPI_H
#define MCA_PINS_PAPI_H

#include "parsec.h"
#include "parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/pins/pins.h"

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_pins_base_component_t parsec_pins_papi_component;
PARSEC_DECLSPEC extern const parsec_pins_module_t parsec_pins_papi_module;
/* static accessor */
mca_base_component_t * pins_papi_static_component(void);

END_C_DECLS

#endif



