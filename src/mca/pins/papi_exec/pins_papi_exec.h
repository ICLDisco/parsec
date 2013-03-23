#ifndef PINS_PAPI_EXEC_H
#define PINS_PAPI_EXEC_H

#include "dague.h"
#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"

#define NUM_EXEC_EVENTS 4

typedef struct papi_exec_info_s {
	int kernel_type;
	int vp_id;
	int th_id;
	int values_len; 
	long long values[NUM_EXEC_EVENTS];
} papi_exec_info_t;

void pins_init_papi_exec(dague_context_t * master_context);
void pins_handle_init_papi_exec(dague_handle_t * handle);
void pins_thread_init_papi_exec(dague_execution_unit_t * exec_unit);

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_papi_exec_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_papi_exec_module;
/* static accessor */
mca_base_component_t * pins_papi_exec_static_component(void);

END_C_DECLS

#endif // PINS_PAPI_EXEC_H
