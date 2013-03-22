#ifndef MCA_PINS_H
#define MCA_PINS_H
/* PaRSEC Performance Instrumentation Callback System */

#include "dague_config.h"
#include "dague_internal.h"
#include "dague/mca/mca.h"

typedef enum PINS_FLAG {
	TASK_SELECT_BEGIN,
	TASK_SELECT_END,
	EXEC_BEGIN,
	EXEC_END,
	/* what follows are
	 * Special Events. 
	 * They do not necessarily
	 * obey the 'exec unit, exec context'
	 * contract. 
	 */
	THREAD_INIT,
	THREAD_FINI,
	HANDLE_INIT,
	HANDLE_FINI,
	/* inactive (no call in source code)
	PARSEC_SCHEDULED,
	PARSEC_PROLOGUE,
	PARSEC_RELEASE,
	 */
	/* this one is not an event at all */
	PINS_FLAG_COUNT
} PINS_FLAG;

typedef void (parsec_pins_callback)(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data); 

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
#define DAGUE_PINS_BASE_VERSION_2_0_0 \
    MCA_BASE_VERSION_2_0_0, \
    "pins", 2, 0, 0

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
void parsec_pins(PINS_FLAG method_flag, dague_execution_unit_t * exec_unit, 
                 dague_execution_context_t * task, void * data);

void pins_disable_registration(int disable);

parsec_pins_callback * pins_register_callback(PINS_FLAG method_flag, parsec_pins_callback * cb);

parsec_pins_callback * pins_unregister_callback(PINS_FLAG method_flag);

#ifdef PINS_ENABLE

#define PINS(method_flag, exec_unit, task, data)                        \
	parsec_pins(method_flag, exec_unit, task, data)
#define PINS_DISABLE_REGISTRATION(boolean) \
	pins_disable_registration(boolean)
#define PINS_REGISTER(method_flag, cb)                                  \
	pins_register_callback(method_flag, cb)
#define PINS_UNREGISTER(method_flag)                                    \
	pins_unregister_callback(method_flag)
#define PINS_INIT(master_context) \
	pins_init(master_context)
#define PINS_THREAD_INIT(exec_unit) \
	pins_thread_init(exec_unit)
#define PINS_HANDLE_INIT(dague_handle) \
	pins_handle_init(dague_handle)
#define PINS_THREAD_FINI(exec_unit) \
	pins_thread_fini(exec_unit)
#define PINS_HANDLE_FINI(dague_handle) \
	pins_handle_fini(dague_handle)

#else // NOT PINS_ENABLE

#define PINS(method_flag, exec_unit, task, data)    \
	do {} while (0)
#define PINS_DISABLE_REGISTRATION(boolean) \
	do {} while(0)
#define PINS_REGISTER(method_flag, cb)                                  \
	do {} while (0)
#define PINS_UNREGISTER(method_flag)                                    \
	do {} while (0)
#define PINS_INIT(master_context) \
	do {} while (0)
#define PINS_THREAD_INIT(exec_unit) \
	do {} while (0)
#define PINS_HANDLE_INIT(dague_handle) \
	do {} while (0)
#define PINS_THREAD_FINI(exec_unit) \
	do {} while (0)
#define PINS_HANDLE_FINI(dague_handle) \
	do {} while (0)

#endif // PINS_ENABLE

#endif // PINS_H
