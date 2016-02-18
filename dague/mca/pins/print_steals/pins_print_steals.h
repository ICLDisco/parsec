#ifndef PINS_PRINT_STEALS_H
#define PINS_PRINT_STEALS_H

#include "dague.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"

#define NUM_SELECT_EVENTS 2
#define SYSTEM_QUEUE_VP -2

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_print_steals_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_print_steals_module;
/* static accessor */
mca_base_component_t * pins_print_steals_static_component(void);

END_C_DECLS

#endif
