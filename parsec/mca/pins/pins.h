/*
 * Copyright (c) 2009-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 */

#ifndef MCA_PINS_H
#define MCA_PINS_H
/* PaRSEC Performance Instrumentation Callback System */

#include "parsec/runtime.h"
#include "parsec/mca/mca.h"

#define PARSEC_PINS_SEPARATOR ";"

struct parsec_pins_next_callback_s;
typedef void (*parsec_pins_callback)(struct parsec_execution_stream_s*   es,
                                     struct parsec_task_s*               task,
                                     struct parsec_pins_next_callback_s* cb_data);

typedef struct parsec_pins_next_callback_s {
    parsec_pins_callback                cb_func;
    struct parsec_pins_next_callback_s* cb_data;
} parsec_pins_next_callback_t;

typedef enum PINS_FLAG {
    SELECT_BEGIN = 0,    // called before scheduler begins looking for an available task
    SELECT_END,          // called after scheduler has finished looking for an available task
    PREPARE_INPUT_BEGIN,
    PREPARE_INPUT_END,
    RELEASE_DEPS_BEGIN,
    RELEASE_DEPS_END,
    ACTIVATE_CB_BEGIN,
    ACTIVATE_CB_END,
    DATA_FLUSH_BEGIN,
    DATA_FLUSH_END,
    EXEC_BEGIN,          // called before thread executes a task
    EXEC_END,            // called before thread executes a task
    COMPLETE_EXEC_BEGIN, // called before scheduler adds a newly-enabled task
    COMPLETE_EXEC_END,   // called after scheduler adds a newly-enabled task
    /* what follows are Special Events. They do not necessarily
     * obey the 'exec unit, exec context' contract.
     */
    THREAD_INIT,         // Provided as an option for modules to run work during thread init without using the MCA module registration system.
    THREAD_FINI,         // Similar to above, for thread finalization.
    /* inactive but tentatively planned (no current call in PaRSEC runtime)
     PARSEC_SCHEDULED,
     PARSEC_PROLOGUE,
     PARSEC_RELEASE,
     */
    /* PINS_FLAG_COUNT is not an event at all */
    PINS_FLAG_COUNT
} PINS_FLAG;

BEGIN_C_DECLS

/*
 * Structures for pins components
 */
struct parsec_pins_base_component_2_0_0 {
    mca_base_component_2_0_0_t base_version;
    mca_base_component_data_2_0_0_t base_data;
};

typedef struct parsec_pins_base_component_2_0_0 parsec_pins_base_component_2_0_0_t;
typedef struct parsec_pins_base_component_2_0_0 parsec_pins_base_component_t;

/*
 * Structure for sched modules
 */

typedef void (*parsec_pins_base_module_init_fn_t)(struct parsec_context_s * master);
typedef void (*parsec_pins_base_module_fini_fn_t)(struct parsec_context_s * master);
typedef void (*parsec_pins_base_module_taskpool_init_fn_t)(struct parsec_taskpool_s * tp);
typedef void (*parsec_pins_base_module_taskpool_fini_fn_t)(struct parsec_taskpool_s * tp);
typedef void (*parsec_pins_base_module_thread_init_fn_t)(struct parsec_execution_stream_s * es);
typedef void (*parsec_pins_base_module_thread_fini_fn_t)(struct parsec_execution_stream_s * es);

struct parsec_pins_base_module_1_0_0_t {
    parsec_pins_base_module_init_fn_t          init;
    parsec_pins_base_module_fini_fn_t          fini;
    parsec_pins_base_module_taskpool_init_fn_t taskpool_init;
    parsec_pins_base_module_taskpool_fini_fn_t taskpool_fini;
    parsec_pins_base_module_thread_init_fn_t   thread_init;
    parsec_pins_base_module_thread_fini_fn_t   thread_fini;
};

typedef struct parsec_pins_base_module_1_0_0_t parsec_pins_base_module_1_0_0_t;
typedef struct parsec_pins_base_module_1_0_0_t parsec_pins_base_module_t;

typedef struct {
    const parsec_pins_base_component_t *component;
    parsec_pins_base_module_t     module;
} parsec_pins_module_t;

/*
 * Macro for use in components that are of type pins, version 2.0.0
 */
#define PARSEC_PINS_BASE_VERSION_2_0_0           \
    MCA_BASE_VERSION_2_0_0,                     \
        "pins", 2, 0, 0

END_C_DECLS

/*
 These functions should each be called once at the appropriate lifecycle of the PaRSEC Context
 except that handle functions should be called once per taskpool, and thread functions once per thread
 */
void pins_init(struct parsec_context_s * master);
void pins_fini(struct parsec_context_s * master);
void pins_taskpool_init(struct parsec_taskpool_s * tp);
void pins_taskpool_fini(struct parsec_taskpool_s * tp);
void pins_thread_init(struct parsec_execution_stream_s * es);
void pins_thread_fini(struct parsec_execution_stream_s * es);

/*
 the following functions are intended for public use wherever they are necessary
 */
void parsec_pins_instrument(struct parsec_execution_stream_s* es,
                            PINS_FLAG method_flag,
                            struct parsec_task_s* task);

void parsec_pins_disable_registration(int disable);

int parsec_pins_is_module_enabled(char * module_name);

int parsec_pins_register_callback(struct parsec_execution_stream_s* es,
                           PINS_FLAG method_flag,
                           parsec_pins_callback cb,
                           parsec_pins_next_callback_t* cb_data);

int parsec_pins_unregister_callback(struct parsec_execution_stream_s* es,
                             PINS_FLAG method_flag,
                             parsec_pins_callback cb,
                             parsec_pins_next_callback_t** cb_data);

#ifdef PINS_ENABLE

#define PINS(unit, method_flag, task)         \
    parsec_pins_instrument(unit, method_flag, task)
#define PINS_DISABLE_REGISTRATION(boolean)      \
    parsec_pins_disable_registration(boolean)
#define PINS_REGISTER(unit, method_flag, cb, data)       \
    parsec_pins_register_callback(unit, method_flag, cb, data)
#define PINS_UNREGISTER(unit, method_flag, cb, pdata)     \
    parsec_pins_unregister_callback(unit, method_flag, cb, pdata)
#define PINS_INIT(master_context)               \
    pins_init(master_context)
#define PINS_FINI(master_context)               \
    pins_fini(master_context)
#define PINS_THREAD_INIT(exec_unit)             \
    pins_thread_init(exec_unit)
#define PINS_TASKPOOL_INIT(parsec_tp)           \
    pins_taskpool_init(parsec_tp)
#define PINS_THREAD_FINI(exec_unit)             \
    pins_thread_fini(exec_unit)
#define PINS_TASKPOOL_FINI(parsec_tp)           \
    pins_taskpool_fini(parsec_tp)

#else // NOT PINS_ENABLE

#define PINS(method_flag, exec_unit, task)      \
    do {} while (0)
#define PINS_DISABLE_REGISTRATION(boolean)      \
    do {} while(0)
#define PINS_REGISTER(method_flag, cb, data)    \
    do {} while (0)
#define PINS_UNREGISTER(method_flag, cb, data)  \
    do {} while (0)
#define PINS_INIT(master_context)               \
    do {} while (0)
#define PINS_FINI(master_context)               \
    do {} while (0)
#define PINS_THREAD_INIT(exec_unit)             \
    do {} while (0)
#define PINS_TASKPOOL_INIT(parsec_tp)           \
    do {} while (0)
#define PINS_THREAD_FINI(exec_unit)             \
    do {} while (0)
#define PINS_TASKPOOL_FINI(parsec_tp)           \
    do {} while (0)

#endif // PINS_ENABLE

#endif // PINS_H
