#include <errno.h>
#include <stdio.h>
#include "dague_config.h"
#include "dague/mca/pins/pins.h"
#include "pins_task_profiler.h"
#include "profiling.h"
#include "execution_unit.h"

/* init functions */
static void pins_init_task_profiler(dague_context_t * master_context);
static void pins_fini_task_profiler(dague_context_t * master_context);

/* PINS callbacks */
static void task_profiler_exec_count_begin(dague_execution_unit_t * exec_unit, 
										   dague_execution_context_t * exec_context, 
										   void * data);
static void task_profiler_exec_count_end(dague_execution_unit_t * exec_unit, 
										 dague_execution_context_t * exec_context, 
										 void * data);

const dague_pins_module_t dague_pins_task_profiler_module = {
    &dague_pins_task_profiler_component,
    {
	    pins_init_task_profiler,
	    pins_fini_task_profiler,
	    NULL,
	    NULL,
		NULL,
		NULL
    }
};

static parsec_pins_callback * exec_begin_prev; /* courtesy calls to previously-registered cbs */
static parsec_pins_callback * exec_end_prev;

static void pins_init_task_profiler(dague_context_t * master_context) {
	(void)master_context;

	exec_begin_prev = PINS_REGISTER(EXEC_BEGIN, task_profiler_exec_count_begin);
	exec_end_prev   = PINS_REGISTER(EXEC_END, task_profiler_exec_count_end);
}

static void pins_fini_task_profiler(dague_context_t * master_context) {
	(void)master_context;
	// replace original registrants
	PINS_REGISTER(EXEC_BEGIN, exec_begin_prev);
	PINS_REGISTER(EXEC_END,   exec_end_prev);
}

/*
  PINS CALLBACKS
*/

static void task_profiler_exec_count_begin(dague_execution_unit_t * exec_unit, 
										   dague_execution_context_t * exec_context, 
										   void * data) {
	if (NULL != exec_context->dague_handle->profiling_array)
		dague_profiling_trace(exec_unit->eu_profile, 
							  exec_context->dague_handle->profiling_array[2 * exec_context->function->function_id],
							  (*exec_context->function->key
								  )(exec_context->dague_handle, exec_context->locals), 
							  exec_context->dague_handle->handle_id, 
							  (void *)NULL);
	// keep the contract with the previous registrant
	if (exec_begin_prev != NULL) {
		(*exec_begin_prev)(exec_unit, exec_context, data);
	}
}

static void task_profiler_exec_count_end(dague_execution_unit_t * exec_unit, 
										 dague_execution_context_t * exec_context, 
										 void * data) {
	if (NULL != exec_context->dague_handle->profiling_array)
		dague_profiling_trace(exec_unit->eu_profile, 
							  exec_context->dague_handle->profiling_array[1 + 2 * exec_context->function->function_id],
							  (*exec_context->function->key
								  )(exec_context->dague_handle, exec_context->locals), 
							  exec_context->dague_handle->handle_id, 
							  (void *)NULL);
	// keep the contract with the previous registrant
	if (exec_end_prev != NULL) {
		(*exec_end_prev)(exec_unit, exec_context, data);
	}
}
