#ifndef MCA_PINS_THREAD_PAPI_SOCKET_H
#define MCA_PINS_THREAD_PAPI_SOCKET_H

#include "dague.h"
#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"

#define NUM_THREAD_PAPI_SOCKET_EVENTS 4

void pins_init_thread_papi_socket(dague_context_t * master_context);

void start_thread_papi_socket(dague_execution_unit_t * exec_unit, 
                              dague_execution_context_t * exec_context, void * data);
void stop_thread_papi_socket(dague_execution_unit_t * exec_unit, 
                             dague_execution_context_t * exec_context, void * data);

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_thread_papi_socket_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_thread_papi_socket_module;
/* static accessor */
mca_base_component_t * pins_thread_papi_socket_static_component(void);

END_C_DECLS

#endif



