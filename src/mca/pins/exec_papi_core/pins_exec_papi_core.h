#ifndef EXEC_PAPI_CORE_H
#define EXEC_PAPI_CORE_H

#include "dague.h"
#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"

#define NUM_EXEC_PAPI_CORE_EVENTS 4

typedef struct pins_exec_papi_core_info_s {
	int kernel_type;
	int vp_id;
	int th_id;
	int values_len;
	long long values[NUM_EXEC_PAPI_CORE_EVENTS];
} pins_exec_papi_core_info_t;

void pins_init_exec_papi_core(dague_context_t * master_context);
void pins_handle_init_exec_papi_core(dague_handle_t * handle);
void pins_thread_init_exec_papi_core(dague_execution_unit_t * exec_unit);

void start_exec_papi_core_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void stop_exec_papi_core_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_exec_papi_core_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_exec_papi_core_module;
/* static accessor */
mca_base_component_t * pins_exec_papi_core_static_component(void);

END_C_DECLS

#endif // EXEC_PAPI_CORE_H
