#ifndef PARSEC_INSTRUMENT_H
#define PARSEC_INSTRUMENT_H

#include "dague_internal.h"

typedef enum INSTRUMENT_FLAG {
	SCHED_INIT,
	SCHED_FINI,
	SCHED_STEAL,
	PARSEC_SCHEDULED,
	PARSEC_PROLOGUE,
	PARSEC_BODY,
	PARSEC_RELEASE,
	A_COUNT_NOT_A_FLAG
} INSTRUMENT_FLAG;

typedef void (parsec_instrument_callback)(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data); 


void parsec_instrument(INSTRUMENT_FLAG method_flag, dague_execution_unit_t * exec_unit, 
                       dague_execution_context_t * task, void * data);

parsec_instrument_callback * register_instrument_callback(INSTRUMENT_FLAG method_flag, parsec_instrument_callback * cb);

parsec_instrument_callback * unregister_instrument_callback(INSTRUMENT_FLAG method_flag);

#define PARSEC_INSTR

#ifdef PARSEC_INSTR
#define PARSEC_INSTRUMENT(method_flag, exec_unit, task, data)    \
	parsec_instrument(method_flag, exec_unit, task, data);
#else
#define PARSEC_INSTRUMENT(method_flag, exec_unit, task, data)    \
	do {} while (0);
#error "whynot"
#endif

#endif
