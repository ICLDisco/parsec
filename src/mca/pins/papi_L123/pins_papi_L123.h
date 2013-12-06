#ifndef PINS_PAPI_L123_H
#define PINS_PAPI_L123_H

#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"
#include "dague.h"

/* Use the following event names as identifiers for the Python parser.
 * e.g., PAPI_L12_EXEC will be parsed as having two event values,
 * the first being L1 misses, the second being L2 misses.
 * PAPI_CORE_EXEC, CORE_SEL, etc. are the generic terms, and will
 * be parsed generically as having 3 total values.
 *
 * Define your own parsers in pbp_info_parser.pxi.
 */
#define PAPI_CORE_PROF_EVT_NAME_SOCKET     "PAPI_SOCKET_DS_L2_ISP"
#define PAPI_CORE_PROF_EVT_NAME_EXEC       "PAPI_CORE_EXEC_DS_L2_ISP"
#define PAPI_CORE_PROF_EVT_NAME_SEL        "PAPI_CORE_SEL_DS_L2_ISP"
#define PAPI_CORE_PROF_EVT_NAME_PREP       "PAPI_CORE_PREP_DS_L2_ISP"
#define PAPI_CORE_PROF_EVT_NAME_COMPL      "PAPI_CORE_COMPL_DS_L2_ISP"

/* enable or disable different phases depending on
 * how many/which events you want per task */
#define ENABLE_EXEC 1
#define ENABLE_SELECT 1
#define ENABLE_PREP 1
#define ENABLE_COMPL 1
#define ENABLE_SOCKET 1

/* reorder the following, but there's no need to delete. */
#define NUM_CORE_EVENTS 3
#define PAPI_CORE_NATIVE_EVENT_ARRAY {                                  \
    "DISPATCH_STALL_FOR_LS_FULL",                                       \
        "L2_CACHE_MISS:DATA",                                           \
        "INEFFECTIVE_SW_PREFETCHES:SW_PREFETCH_HIT_IN_L1",              \
        }
/*
        "L3_CACHE_MISSES:ANY_READ",                                     \
        "L2_CACHE_MISS:HW_PREFETCH_FROM_DC",                            \
        "MAB_REQUESTS",                                                 \
        "INEFFECTIVE_SW_PREFETCHES:SW_PREFETCH_HIT_IN_L1",              \
        "DISPATCH_STALL_FOR_RESERVATION_STATION_FULL",                  \
        "READ_REQUEST_TO_L3_CACHE",                                     \
        "MAB_WAIT_CYCLES",                                              \
        "DISPATCH_STALL_FOR_SEGMENT_LOAD",                              \
        "INEFFECTIVE_SW_PREFETCHES:SW_PREFETCH_HIT_IN_L2",              \
        "CACHE_BLOCK:ALL",                                              \
        "DATA_CACHE_REFILLS:L2_EXCLUSIVE",                              \
        "DATA_PREFETCHES",                                              \
        "MEMORY_REQUESTS",                                              \
        "SYSTEM_READ_RESPONSES:EXCLUSIVE:OWNED:SHARED:MODIFIED",        \
        "LLC-LOAD-MISSES",                                              \
        "MISALIGNED_ACCESSES",                                          \
        }
 */
#define NUM_SOCKET_EVENTS 0
#define PAPI_SOCKET_NATIVE_EVENT_ARRAY {} /*{"L3_CACHE_MISSES:ANY_READ"}*/

#define SYSTEM_QUEUE_VP -2

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
    unsigned long long int selection_time;
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
