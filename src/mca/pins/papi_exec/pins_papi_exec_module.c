#include "dague_config.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "pins_papi_exec.h"
#include <papi.h>
#include <stdio.h>
#include "profiling.h"
#include "execution_unit.h"

/* these should eventually be runtime-configurable */
static int exec_events[NUM_EXEC_EVENTS] = {PAPI_L1_DCM, PAPI_L2_DCH, PAPI_L2_DCM, PAPI_L2_DCA};

static void pins_init_papi_exec(dague_context_t * master_context);
static void pins_fini_papi_exec(dague_context_t * master_context);
static void pins_thread_init_papi_exec(dague_execution_unit_t * exec_unit);

const dague_pins_module_t dague_pins_papi_exec_module = {
    &dague_pins_papi_exec_component,
    {
	    pins_init_papi_exec,
	    pins_fini_papi_exec,
	    NULL,
	    NULL,
	    pins_thread_init_papi_exec,
	    NULL
    }
};

static void start_papi_exec_count(dague_execution_unit_t * exec_unit, 
                                  dague_execution_context_t * exec_context, 
                                  void * data);
static void stop_papi_exec_count(dague_execution_unit_t * exec_unit, 
                                 dague_execution_context_t * exec_context, 
                                 void * data);

static parsec_pins_callback * exec_begin_prev; // courtesy calls to previously-registered cbs
static parsec_pins_callback * exec_end_prev;

static int papi_socket_enabled; // defaults to false

static int pins_prof_papi_exec_begin, pins_prof_papi_exec_end;

static void pins_init_papi_exec(dague_context_t * master_context) {
	(void)master_context;
	exec_begin_prev = PINS_REGISTER(EXEC_BEGIN, start_papi_exec_count);
	exec_end_prev   = PINS_REGISTER(EXEC_END, stop_papi_exec_count);
	dague_profiling_add_dictionary_keyword("PINS_EXEC", "fill:#00FF00",
	                                       sizeof(papi_exec_info_t), NULL,
	                                       &pins_prof_papi_exec_begin, &pins_prof_papi_exec_end);

	papi_socket_enabled = is_pins_module_enabled("papi_socket");
}

static void pins_fini_papi_exec(dague_context_t * master_context) {
	(void)master_context;
	// replace original registrant
	PINS_REGISTER(EXEC_BEGIN, exec_begin_prev);
	PINS_REGISTER(EXEC_END,   exec_end_prev);
}

static void pins_thread_init_papi_exec(dague_execution_unit_t * exec_unit) {
	int rv = 0;
	if (!papi_socket_enabled || 
		exec_unit->th_id % CORES_PER_SOCKET != WHICH_CORE_IN_SOCKET) {
		exec_unit->papi_eventsets[EXEC_SET] = PAPI_NULL;
		if ((rv = PAPI_create_eventset(&exec_unit->papi_eventsets[EXEC_SET])) != PAPI_OK)
            printf("papi_exec_module.c, pins_thread_init_papi_exec: "
				   "thread %d couldn't create the PAPI event set "
				   "to measure L1/L2 misses; ERROR: %s\n", 
				   exec_unit->th_id, PAPI_strerror(rv));
		if ((rv = PAPI_add_events(exec_unit->papi_eventsets[EXEC_SET], 
								  exec_events, NUM_EXEC_EVENTS)) 
		    != PAPI_OK)
			printf("papi_exec.c, pins_thread_init_papi_exec: thread %d failed to add "
			       "exec events to EXEC eventset; ERROR: %s\n", 
				   exec_unit->th_id, PAPI_strerror(rv));
	}
}

static void start_papi_exec_count(dague_execution_unit_t * exec_unit, 
                                  dague_execution_context_t * exec_context, 
                                  void * data) {
	int rv = PAPI_OK;
	if (!papi_socket_enabled || 
		exec_unit->th_id % CORES_PER_SOCKET != WHICH_CORE_IN_SOCKET) {
		if ((rv = PAPI_start(exec_unit->papi_eventsets[EXEC_SET])) != PAPI_OK) {
			printf("papi_exec.c, start_papi_exec_count: thread %d can't start "
			       "exec event counters! %s\n", 
			       exec_unit->th_id, PAPI_strerror(rv));
		}
		else { // handle, 
			dague_profiling_trace(exec_unit->eu_profile, pins_prof_papi_exec_begin, 
			                      (*exec_context->function->key
			                       )(exec_context->dague_handle, exec_context->locals), 
			                      exec_context->dague_handle->handle_id, NULL);

		}
	}
	// keep the contract with the previous registrant
	if (exec_begin_prev != NULL) {
		(*exec_begin_prev)(exec_unit, exec_context, data);
	}
}

static void stop_papi_exec_count(dague_execution_unit_t * exec_unit, 
                                 dague_execution_context_t * exec_context, 
                                 void * data) {
	(void)exec_context;
	(void)data;
	long long int values[NUM_EXEC_EVENTS];
	int rv = PAPI_OK;
	if (!papi_socket_enabled ||
		exec_unit->th_id % CORES_PER_SOCKET != WHICH_CORE_IN_SOCKET) {
		if ((rv = PAPI_stop(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
			printf("papi_exec_module.c, stop_papi_exec_count: "
				   "thread %d can't stop exec event counters! "
				   "ERROR: %s\n", 
			       exec_unit->th_id, PAPI_strerror(rv));
		}
		else {
			papi_exec_info_t info;
			info.kernel_type = exec_context->function->function_id;
			strncpy(info.kernel_name, exec_context->function->name, KERNEL_NAME_SIZE - 1);
			info.kernel_name[KERNEL_NAME_SIZE - 1] = '\0';
			info.vp_id = exec_unit->virtual_process->vp_id;
			info.th_id = exec_unit->th_id;
			for(int i = 0; i < NUM_EXEC_EVENTS; i++) 
				info.values[i] = values[i];

			/* not *necessary*, but perhaps better for compatibility
			 * with the Python dbpreader script in the long term,
			 * since this will allow the reading of different structs.
			 * presumably, a 'generic' Cython info reader could be created
			 * that allows a set of ints and a set of long longs
			 * to be automatically read if both lengths are included,
			 * e.g. struct { int num_ints; int; int; int; int num_lls;
			 * ll; ll; ll; ll; ll; ll } - the names could be assigned
			 * after the fact by a knowledgeable end user */
			info.values_len = NUM_EXEC_EVENTS; 

			dague_profiling_trace(exec_unit->eu_profile, pins_prof_papi_exec_end, 
			                      (*exec_context->function->key
			                       )(exec_context->dague_handle, exec_context->locals), 
			                      exec_context->dague_handle->handle_id, 
			                      (void *)&info);

		}
	}
	// keep the contract with the previous registerer
	if (exec_end_prev != NULL) {
		(*exec_end_prev)(exec_unit, exec_context, data);
	}
}

