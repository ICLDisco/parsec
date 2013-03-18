#ifndef PINS_H
#define PINS_H
/* PaRSEC Performance Instrumentation Callback System */

#include "dague_config.h"
#include "dague_internal.h"

typedef enum PINS_FLAG {
	TASK_SELECT_BEGIN,
	TASK_SELECT_FINI,
	EXEC_BEGIN,
	EXEC_FINI,
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
	PARSEC_SCHEDULED,
	PARSEC_PROLOGUE,
	PARSEC_RELEASE,
	/* this one is not an event at all */
	A_COUNT_NOT_A_FLAG
} PINS_FLAG;

typedef void (parsec_pins_callback)(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data); 

void pins_init(dague_context_t * master_context); // impl provided by CMake
void pins_handle_init(dague_handle_t * handle);   // impl provided by CMake
void pins_thread_init(dague_execution_unit_t * exec_unit);   // impl provided by CMake
void pins_thread_fini(dague_execution_unit_t * exec_unit);

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

#else // !PINS_ENABLE

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
