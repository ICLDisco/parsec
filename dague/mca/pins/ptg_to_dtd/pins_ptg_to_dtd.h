#ifndef PINS_PTG_TO_DTD_H
#define PINS_PTG_TO_DTD_H

#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"
#include "dague.h"

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_ptg_to_dtd_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_ptg_to_dtd_module;
/* static accessor */
mca_base_component_t * pins_ptg_to_dtd_static_component(void);

END_C_DECLS

#endif // PINS_PTG_TO_DTD_H
