#ifndef MCA_PINS_PAPI_SOCKET_H
#define MCA_PINS_PAPI_SOCKET_H

#include "dague.h"
#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"

#define NUM_SOCKET_EVENTS 3
#define KERNEL_NAME_SIZE 9

typedef struct papi_socket_info_s {
	int kernel_type;
	char kernel_name[KERNEL_NAME_SIZE];
	int vp_id;
	int th_id;
	int values_len; 
	long long values[NUM_SOCKET_EVENTS];
} papi_socket_info_t;

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



