#ifndef PINS_PAPI_L123_H
#define PINS_PAPI_L123_H

#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"
#include "dague.h"

#define NUM_L12_EVENTS 2
#define SYSTEM_QUEUE_VP -2

typedef struct papi_L123_info_s {
	int vp_id;
	int th_id;
	long long L1_misses;
	long long L2_misses;
	long long L3_misses; // most interesting
} papi_L123_info_t;

typedef struct papi_L12_exec_info_s {
	int kernel_type;
	int vp_id;
	int th_id;
	long long L1_misses;
	long long L2_misses;
	long long L3_misses; // unnecessary but potentially interesting
} papi_L12_exec_info_t;

typedef struct papi_L12_select_info_s {
	int kernel_type;
	int vp_id;
	int th_id;
	int victim_vp_id;
	int victim_th_id;
	unsigned long long int starvation;
	unsigned long long int exec_context;
	long long L1_misses;
	long long L2_misses;
	long long L3_misses; // unnecessary but potentially interesting
} papi_L12_select_info_t;

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_papi_L123_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_papi_L123_module;
/* static accessor */
mca_base_component_t * pins_papi_exec_static_component(void);

END_C_DECLS

#endif // PINS_PAPI_EXEC_H
