#ifndef CACHEMISS_H
#define CACHEMISS_H
#include "dague.h"

#define NUM_EXEC_EVENTS 4

typedef struct pins_cachemiss_info_s {
	int kernel_type;
	int vp_id;
	int th_id;
	int values_len;
	long long values[NUM_EXEC_EVENTS];
} pins_cachemiss_info_t;

void pins_init_cachemiss(dague_context_t * master_context);
void pins_handle_init_cachemiss(dague_handle_t * handle);
void pins_thread_init_cachemiss(dague_execution_unit_t * exec_unit);

void start_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void stop_papi_exec_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

#endif // CACHEMISS_H
