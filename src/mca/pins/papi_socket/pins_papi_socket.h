#ifndef MCA_PINS_PAPI_SOCKET_H
#define MCA_PINS_PAPI_SOCKET_H

#include "dague.h"
#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"

#define NUM_SOCKET_EVENTS 4

void pins_init_papi_socket(dague_context_t * master_context);

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_papi_socket_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_papi_socket_module;
/* static accessor */
mca_base_component_t * pins_papi_socket_static_component(void);

END_C_DECLS

#endif



