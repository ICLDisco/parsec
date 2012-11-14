#include <stdlib.h>
#include "instrument.h"
#include "debug.h"

parsec_instrument_callback * instrument_array[A_COUNT_NOT_A_FLAG] = { 0 };

static void empty_callback(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data);

void parsec_instrument(INSTRUMENT_FLAG method_flag, 
                       dague_execution_unit_t * exec_unit,
                       dague_execution_context_t * task, 
                       void * data) {
	(*(instrument_array[method_flag]))(exec_unit, task, data);
}

/**
 The behavior of the PARSEC_INSTRUMENT system is undefined if 
 register_instrument_callback is not called at least once before 
 any call to parsec_instrument.
 */
parsec_instrument_callback * register_instrument_callback(INSTRUMENT_FLAG method_flag, parsec_instrument_callback * cb) {
	if (!instrument_array[0]) {
		int i = 0;
		for (; i < A_COUNT_NOT_A_FLAG; i++) {
			if (instrument_array[i] == NULL)
				instrument_array[i] = &empty_callback;
		}
	}
	assert(cb != NULL);
	if (method_flag >= 0 && method_flag < A_COUNT_NOT_A_FLAG) {
		parsec_instrument_callback * prev = instrument_array[method_flag];
		instrument_array[method_flag] = cb;
		return prev;
	}
	else {
		DEBUG(("WARNING: Attempted to use the invalid flag %d with PaRSEC Instrumentation!\n", method_flag));
	}
	return NULL;
}

parsec_instrument_callback * unregister_instrument_callback(INSTRUMENT_FLAG method_flag) {
	if (method_flag >= PARSEC_SCHEDULED && method_flag < A_COUNT_NOT_A_FLAG) {
		parsec_instrument_callback * prev = instrument_array[method_flag];
		instrument_array[method_flag] = &empty_callback;
		return prev;
	}
	else {
		DEBUG(("WARNING: Attempted to use the invalid flag %d with PaRSEC Instrumentation!\n", method_flag));
	}
	return NULL;
}

static void empty_callback(dague_execution_unit_t * exec_unit, dague_execution_context_t * task, void * data) {
	// do nothing
	(void) exec_unit;
	(void) task;
	(void) data;
	printf("doing nothing\n");
}
