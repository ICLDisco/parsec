#ifndef PINS_H
#define PINS_H
/* PaRSEC Performance Instrumentation Callback System */

#include "dague_internal.h"

typedef enum PINS_FLAG {
	SCHED_STEAL,
	TASK_SELECT_BEGIN,
	TASK_SELECT_FINI,
	PARSEC_SCHEDULED,
	PARSEC_PROLOGUE,
	EXEC_BEGIN,
	EXEC_FINI,
	PARSEC_RELEASE,
	A_COUNT_NOT_A_FLAG
} PINS_FLAG;

typedef void (parsec_pins_callback)(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data); 

void pins_init(dague_context_t * master_context); // impl provided by CMake
void pins_handle_init(dague_handle_t * handle);   // impl provided by CMake

void parsec_pins(PINS_FLAG method_flag, dague_execution_unit_t * exec_unit, 
                 dague_execution_context_t * task, void * data);

parsec_pins_callback * pins_register_callback(PINS_FLAG method_flag, parsec_pins_callback * cb);

parsec_pins_callback * pins_unregister_callback(PINS_FLAG method_flag);

void pins_construct(void); // pins_init.c is generated and provides the implementation

#ifdef PINS_ENABLE

#define PINS(method_flag, exec_unit, task, data)                        \
	parsec_pins(method_flag, exec_unit, task, data)
#define PINS_REGISTER(method_flag, cb)                                  \
	pins_register_callback(method_flag, cb)
#define PINS_UNREGISTER(method_flag)                                    \
	pins_unregister_callback(method_flag)

#else // !PINS_ENABLE

#define PINS(method_flag, exec_unit, task, data)    \
	do {} while (0)
#define PINS_REGISTER(method_flag, cb)                                  \
	do {} while (0)
#define PINS_UNREGISTER(method_flag)                                    \
	do {} while (0)

#endif // PINS_ENABLE

#endif // PINS_H
