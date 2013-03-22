#ifndef STEALS_H
#define STEALS_H
#include "dague.h"

#define NUM_TASK_SELECT_EVENTS 2
#define SYSTEM_QUEUE_VP -2

typedef struct pins_task_select_info_s {
	int kernel_type;
	int vp_id;
	int th_id;
	int victim_vp_id;
	int victim_th_id;
	unsigned long long int exec_context;
	int values_len;
	long long values[NUM_TASK_SELECT_EVENTS];
} pins_task_select_info_t;

void pins_init_steals(dague_context_t * master_context);
void pins_thread_init_steals(dague_execution_unit_t * exec_unit);
void pins_fini_steals(dague_context_t * master_context);

void start_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);
void stop_papi_steal_count(dague_execution_unit_t * exec_unit, dague_execution_context_t * exec_context, void * data);

#endif
