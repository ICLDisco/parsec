#ifndef PINS_ITERATORS_CHECKER_H
#define PINS_ITERATORS_CHECKER_H

#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"
#include "dague.h"

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_iterators_checker_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_iterators_checker_module;
/* static accessor */
mca_base_component_t * pins_iterators_checker_static_component(void);

END_C_DECLS

#endif // PINS_ITERATORS_CHECKER_H
