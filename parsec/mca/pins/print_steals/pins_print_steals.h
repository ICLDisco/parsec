#ifndef PINS_PRINT_STEALS_H
#define PINS_PRINT_STEALS_H

#include "parsec_config.h"
#include "parsec.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/pins/pins.h"

#define NUM_SELECT_EVENTS 2
#define SYSTEM_QUEUE_VP -2

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_pins_base_component_t parsec_pins_print_steals_component;
PARSEC_DECLSPEC extern const parsec_pins_module_t parsec_pins_print_steals_module;
/* static accessor */
mca_base_component_t * pins_print_steals_static_component(void);

END_C_DECLS

#endif
