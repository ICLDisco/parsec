#ifndef PINS_PTG_TO_DTD_H
#define PINS_PTG_TO_DTD_H

#include "parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/pins/pins.h"
#include "parsec.h"

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_pins_base_component_t parsec_pins_ptg_to_dtd_component;
PARSEC_DECLSPEC extern const parsec_pins_module_t parsec_pins_ptg_to_dtd_module;
/* static accessor */
mca_base_component_t * pins_ptg_to_dtd_static_component(void);

END_C_DECLS

#endif // PINS_PTG_TO_DTD_H
