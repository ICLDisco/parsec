#ifndef PINS_ITERATORS_CHECKER_H
#define PINS_ITERATORS_CHECKER_H
/*
 * Copyright (c) 2016-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */


#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/runtime.h"

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_pins_base_component_t parsec_pins_iterators_checker_component;
PARSEC_DECLSPEC extern const parsec_pins_module_t parsec_pins_iterators_checker_module;
/* static accessor */
mca_base_component_t * pins_iterators_checker_static_component(void);

END_C_DECLS

#endif // PINS_ITERATORS_CHECKER_H
