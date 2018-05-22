#ifndef PINS_TASK_PROFILER_H
#define PINS_TASK_PROFILER_H

#include "parsec/parsec_config.h"
#include "parsec/mca/mca.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/runtime.h"

typedef struct task_profiler_info_s {
    int kernel_type;
    int vp_id;
    int th_id;
} task_profiler_info_t;

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
PARSEC_DECLSPEC extern const parsec_pins_base_component_t parsec_pins_task_profiler_component;
PARSEC_DECLSPEC extern const parsec_pins_module_t parsec_pins_task_profiler_module;
/* static accessor */
mca_base_component_t * pins_task_profiler_static_component(void);

END_C_DECLS

#endif // PINS_TASK_PROFILER_H
