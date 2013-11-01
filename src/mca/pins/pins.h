#ifndef MCA_PINS_H
#define MCA_PINS_H
/* PaRSEC Performance Instrumentation Callback System */

#include "dague_config.h"
#include "dague_internal.h"
#include "dague/mca/mca.h"

typedef enum PINS_FLAG {
    SELECT_BEGIN,    // called before scheduler begins looking for an available task
    SELECT_END,      // called after scheduler has finished looking for an available task
    PREPARE_INPUT_BEGIN,
    PREPARE_INPUT_END,
    EXEC_BEGIN,      // called before thread executes a task
    EXEC_END,        // called before thread executes a task
    COMPLETE_EXEC_BEGIN,       // called before scheduler adds a newly-enabled task
    COMPLETE_EXEC_END,         // called after scheduler adds a newly-enabled task
    /* what follows are
     * Special Events.
     * They do not necessarily
     * obey the 'exec unit, exec context'
     * contract.
     */
    THREAD_INIT,     // Provided as an option for modules to run work during thread init without using the MCA module registration system.
    THREAD_FINI,     // Similar to above, for thread finalization.
    HANDLE_INIT,     // Similar to {{{THREAD_INIT}}}, for handle initialization.
    HANDLE_FINI,     // Similar to {{{THREAD_FINI}}}, for handle finalialization.
    /* inactive but tentatively planned (no current call in PaRSEC runtime)
     PARSEC_SCHEDULED,
     PARSEC_PROLOGUE,
     PARSEC_RELEASE,
     */
    /* PINS_FLAG_COUNT is not an event at all */
    PINS_FLAG_COUNT
} PINS_FLAG;

typedef void (parsec_pins_callback)(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data);

void pins_empty_callback(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data);

BEGIN_C_DECLS

/*
 * Structures for pins components
 */

struct dague_pins_base_component_2_0_0 {
    mca_base_component_2_0_0_t base_version;
    mca_base_component_data_2_0_0_t base_data;
};

typedef struct dague_pins_base_component_2_0_0 dague_pins_base_component_2_0_0_t;
typedef struct dague_pins_base_component_2_0_0 dague_pins_base_component_t;

/*
 * Structure for sched modules
 */

typedef void (*dague_pins_base_module_init_fn_t)(dague_context_t * master);
typedef void (*dague_pins_base_module_fini_fn_t)(dague_context_t * master);
typedef void (*dague_pins_base_module_handle_init_fn_t)(dague_handle_t * handle);
typedef void (*dague_pins_base_module_handle_fini_fn_t)(dague_handle_t * handle);
typedef void (*dague_pins_base_module_thread_init_fn_t)(dague_execution_unit_t * exec_unit);
typedef void (*dague_pins_base_module_thread_fini_fn_t)(dague_execution_unit_t * exec_unit);

struct dague_pins_base_module_1_0_0_t {
    dague_pins_base_module_init_fn_t        init;
    dague_pins_base_module_fini_fn_t        fini;
    dague_pins_base_module_handle_init_fn_t handle_init;
    dague_pins_base_module_handle_fini_fn_t handle_fini;
    dague_pins_base_module_thread_init_fn_t thread_init;
    dague_pins_base_module_thread_fini_fn_t thread_fini;
};

typedef struct dague_pins_base_module_1_0_0_t dague_pins_base_module_1_0_0_t;
typedef struct dague_pins_base_module_1_0_0_t dague_pins_base_module_t;

typedef struct {
    const dague_pins_base_component_t *component;
    dague_pins_base_module_t     module;
} dague_pins_module_t;

/*
 * Macro for use in components that are of type pins, version 2.0.0
 */
#define DAGUE_PINS_BASE_VERSION_2_0_0           \
    MCA_BASE_VERSION_2_0_0,                     \
        "pins", 2, 0, 0

END_C_DECLS

/*
 These functions should be each be called once at the appropriate lifecycle of the DAGuE Context
 except that handle functions should be called once per handle, and thread functions once per thread
 */
void pins_init(dague_context_t * master);
void pins_fini(dague_context_t * master);
void pins_handle_init(dague_handle_t * handle);
void pins_handle_fini(dague_handle_t * handle);
void pins_thread_init(dague_execution_unit_t * exec_unit);
void pins_thread_fini(dague_execution_unit_t * exec_unit);

/*
 the following functions are intended for public use wherever they are necessary
 */
void parsec_instrument(PINS_FLAG method_flag, dague_execution_unit_t * exec_unit,
                       dague_execution_context_t * task, void * data);

void pins_disable_registration(int disable);

void pins_enable_modules (const char * const modules[]);

int pins_is_module_enabled(char * module_name);

parsec_pins_callback * pins_register_callback(PINS_FLAG method_flag, parsec_pins_callback * cb);

parsec_pins_callback * pins_unregister_callback(PINS_FLAG method_flag);



#ifdef PINS_ENABLE

#define PINS(method_flag, exec_unit, task, data)                \
    parsec_instrument(method_flag, exec_unit, task, data)
#define PINS_DISABLE_REGISTRATION(boolean)      \
    pins_disable_registration(boolean)
#define PINS_REGISTER(method_flag, cb)          \
    pins_register_callback(method_flag, cb)
#define PINS_UNREGISTER(method_flag)            \
    pins_unregister_callback(method_flag)
#define PINS_INIT(master_context)               \
    pins_init(master_context)
#define PINS_FINI(master_context)               \
    pins_fini(master_context)
#define PINS_THREAD_INIT(exec_unit)             \
    pins_thread_init(exec_unit)
#define PINS_HANDLE_INIT(dague_handle)          \
    pins_handle_init(dague_handle)
#define PINS_THREAD_FINI(exec_unit)             \
    pins_thread_fini(exec_unit)
#define PINS_HANDLE_FINI(dague_handle)          \
    pins_handle_fini(dague_handle)

#else // NOT PINS_ENABLE

#define PINS(method_flag, exec_unit, task, data)        \
    do {} while (0)
#define PINS_DISABLE_REGISTRATION(boolean)      \
    do {} while(0)
#define PINS_REGISTER(method_flag, cb)          \
    do {} while (0)
#define PINS_UNREGISTER(method_flag)            \
    do {} while (0)
#define PINS_INIT(master_context)               \
    do {} while (0)
#define PINS_FINI(master_context)               \
    do {} while (0)
#define PINS_THREAD_INIT(exec_unit)             \
    do {} while (0)
#define PINS_HANDLE_INIT(dague_handle)          \
    do {} while (0)
#define PINS_THREAD_FINI(exec_unit)             \
    do {} while (0)
#define PINS_HANDLE_FINI(dague_handle)          \
    do {} while (0)

#endif // PINS_ENABLE

#endif // PINS_H
