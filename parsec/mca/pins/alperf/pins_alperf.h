#ifndef PINS_ALPERF_H
#define PINS_ALPERF_H

#include "parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/pins/pins.h"
#include "parsec.h"

BEGIN_C_DECLS

/**
 * This module gathers user-selected / programmer-defined statistics into
 * a single structure. That structure is of variable size (decided at runtime),
 * so it is pointed to with a void * (it’s the field pins_alperf_counter_store.counters,
 * and its size is what is returned by pins_alperf_counter_store_size()).
 * The format of that buffer is the following:
     double <date of last read, in seconds, must be updated by the monitoring thread>
     uint16_t <Number of times the following sequence repeats itself>
       char name[8] <name of the selected stat. e.g.: “task\0\0\0\0”, or “flops\0\0\0”>
       uint64_t value0 <value for eu 0>
       uint64_t value1 <value for eu 1>
       …
       uint64_t value<n-1> <value for eu n-1, where n is the number of execution units for the sending process>

 * The structure is not padded at all: it is packed, and all alignment is done by converting
 * to a uintptr_t and adding the right number of bytes with the macro PINS_ALPERF_COUNTER().
 */

#define PINS_ALPERF_COUNTER_NAME_MAX 8

typedef struct {
    char name[PINS_ALPERF_COUNTER_NAME_MAX];
    uint64_t value_per_eu[1];
} pins_alperf_counter_t;

typedef struct {
    size_t    ct_size;
    uint16_t  nb_counters;
    void     *counters;
} pins_alperf_counter_store_t;

PARSEC_DECLSPEC extern pins_alperf_counter_store_t pins_alperf_counter_store;

static inline size_t pins_alperf_counter_store_size(void)
{
    return sizeof(double) + sizeof(uint16_t) + pins_alperf_counter_store.ct_size * pins_alperf_counter_store.nb_counters;
}

#define PINS_ALPERF_DATE       ((double*)pins_alperf_counter_store.counters)
#define PINS_ALPERF_NBCOUNTERS ((uint16_t*)((uintptr_t)pins_alperf_counter_store.counters + sizeof(double)))
#define PINS_ALPERF_COUNTER(i) ((pins_alperf_counter_t*)((uintptr_t)pins_alperf_counter_store.counters + (sizeof(double) + sizeof(uint16_t) + ((i) * pins_alperf_counter_store.ct_size))))

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_pins_base_component_t parsec_pins_alperf_component;
PARSEC_DECLSPEC extern const parsec_pins_module_t parsec_pins_alperf_module;
/* static accessor */
mca_base_component_t * pins_alperf_static_component(void);

END_C_DECLS

#endif // PINS_ALPERF_H
