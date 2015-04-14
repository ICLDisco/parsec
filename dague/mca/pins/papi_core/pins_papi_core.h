#ifndef MCA_PINS_PAPI_CORE_H
#define MCA_PINS_PAPI_CORE_H

#include "dague.h"
#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_papi_core_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_papi_core_module;
/* static accessor */
mca_base_component_t * pins_papi_core_static_component(void);

END_C_DECLS

#endif



