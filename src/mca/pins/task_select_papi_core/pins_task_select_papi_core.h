#ifndef TASK_SELECT_PAPI_CORE_H
#define TASK_SELECT_PAPI_CORE_H

#include "dague.h"
#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"

#define NUM_TASK_SELECT_EVENTS 2
#define SYSTEM_QUEUE_VP -2

typedef struct pins_task_select_info_s {
	int kernel_type;
	int vp_id;
	int th_id;
	int victim_vp_id;
	int victim_th_id;
	unsigned long long int exec_context;
	int values_len;
	long long values[NUM_TASK_SELECT_EVENTS];
} pins_task_select_info_t;

void pins_init_task_select_papi_core(dague_context_t * master_context);
void pins_fini_task_select_papi_core(dague_context_t * master_context);
void pins_thread_init_task_select_papi_core(dague_execution_unit_t * exec_unit);

void start_task_select_papi_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void stop_task_select_papi_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_task_select_papi_core_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_task_select_papi_core_module;
/* static accessor */
mca_base_component_t * pins_task_select_papi_core_static_component(void);

END_C_DECLS

#endif
