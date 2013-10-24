#ifndef PINS_TASK_PROFILER_H
#define PINS_TASK_PROFILER_H

#include "dague_config.h"
#include "dague/mca/mca.h"
#include "dague/mca/pins/pins.h"
#include "dague.h"

typedef struct task_profiler_info_s {
    int kernel_type;
    int vp_id;
    int th_id;
} task_profiler_info_t;

BEGIN_C_DECLS

/**
 * Globally exported variable
 */
DAGUE_DECLSPEC extern const dague_pins_base_component_t dague_pins_task_profiler_component;
DAGUE_DECLSPEC extern const dague_pins_module_t dague_pins_task_profiler_module;
/* static accessor */
mca_base_component_t * pins_task_profiler_static_component(void);

END_C_DECLS

#endif // PINS_TASK_PROFILER_H
