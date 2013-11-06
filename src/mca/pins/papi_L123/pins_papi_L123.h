#ifndef PINS_PAPI_L123_H
#define PINS_PAPI_L123_H

#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"
#include "dague.h"

#define NUM_CORE_EVENTS 3
#define PAPI_CORE_NATIVE_EVENT_ARRAY {          \
        "L2_CACHE_MISS:DATA",                   \
        "L3_CACHE_MISSES:READ_BLOCK_EXCLUSIVE",     \
        "DTLB-LOAD-MISSES",                \
}
#define NUM_SOCKET_EVENTS 0
#define PAPI_SOCKET_NATIVE_EVENT_ARRAY {} /*{"L3_CACHE_MISSES:ANY_READ"}*/

/* other useful events... */
/* "L3_CACHE_MISSES:READ_BLOCK_EXCLUSIVE" */

#define ENABLE_EXEC 1
#define ENABLE_SELECT 0
#define ENABLE_COMPL 0

#define SYSTEM_QUEUE_VP -2

/* Use the following event names as identifiers for the Python parser.
 * e.g., PAPI_L12_EXEC will be parsed as having two event values,
 * the first being L1 misses, the second being L2 misses.
 * PAPI_CORE_EXEC, CORE_SEL, etc. are the generic terms, and will
 * be parsed generically as having 3 total values.
 *
 * Define your own parsers in pbp_info_parser.pxi.
 */
#define PAPI_CORE_PROF_EVT_NAME_SOCKET     "PAPI_SOCKET_23T"
#define PAPI_CORE_PROF_EVT_NAME_EXEC       "PAPI_CORE_EXEC_23T"
#define PAPI_CORE_PROF_EVT_NAME_SEL        "PAPI_CORE_SEL_23T"
#define PAPI_CORE_PROF_EVT_NAME_COMPL      "PAPI_CORE_COMPL_23T"

typedef struct papi_core_socket_info_s {
    long long evt_values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
} papi_core_socket_info_t;

typedef struct papi_core_exec_info_s {
    int kernel_type;
    long long evt_values[NUM_CORE_EVENTS];
} papi_core_exec_info_t;

typedef struct papi_core_select_info_s {
    int kernel_type;
    int victim_vp_id;
    int victim_th_id;
    unsigned long long int starvation;
    unsigned long long int exec_context;
    long long evt_values[NUM_CORE_EVENTS];
} papi_core_select_info_t;

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_papi_L123_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_papi_L123_module;
/* static accessor */
mca_base_component_t * pins_papi_L123_static_component(void);

END_C_DECLS

#endif // PINS_PAPI_L123_H
