#include <errno.h>
#include "dague_config.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "pins_papi_L123.h"
#include <papi.h>
#include <stdio.h>
#include "profiling.h"
#include "execution_unit.h"

static char* L12_events [NUM_L12_EVENTS] = {"L1-DCACHE-LOAD-MISSES",
											"L2_CACHE_MISS:DATA"};
static char* L3_event = "L3_CACHE_MISSES:READ_BLOCK_EXCLUSIVE";

static void pins_init_papi_L123(dague_context_t * master_context);
static void pins_fini_papi_L123(dague_context_t * master_context);
static void pins_thread_init_papi_L123(dague_execution_unit_t * exec_unit);
static void pins_thread_fini_papi_L123(dague_execution_unit_t * exec_unit);

static void read_papi_L12_exec_count_begin(dague_execution_unit_t * exec_unit, 
										   dague_execution_context_t * exec_context, 
										   void * data);
static void read_papi_L12_exec_count_end(dague_execution_unit_t * exec_unit, 
										 dague_execution_context_t * exec_context, 
										 void * data);
static void read_papi_L12_select_count_begin(dague_execution_unit_t * exec_unit, 
											 dague_execution_context_t * exec_context, 
											 void * data);
static void read_papi_L12_select_count_end(dague_execution_unit_t * exec_unit, 
										   dague_execution_context_t * exec_context, 
										   void * data);
static void read_papi_L12_add_count_begin(dague_execution_unit_t * exec_unit, 
										  dague_execution_context_t * exec_context, 
										  void * data);
static void read_papi_L12_add_count_end(dague_execution_unit_t * exec_unit, 
										dague_execution_context_t * exec_context, 
										void * data);

const dague_pins_module_t dague_pins_papi_L123_module = {
    &dague_pins_papi_L123_component,
    {
	    pins_init_papi_L123,
	    pins_fini_papi_L123,
	    NULL,
	    NULL,
	    pins_thread_init_papi_L123,
	    pins_thread_fini_papi_L123
    }
};

static parsec_pins_callback * exec_begin_prev; // courtesy calls to previously-registered cbs
static parsec_pins_callback * exec_end_prev;
static parsec_pins_callback * select_begin_prev; // courtesy calls to previously-registered cbs
static parsec_pins_callback * select_end_prev;
static parsec_pins_callback * add_begin_prev; // courtesy calls to previously-registered cbs
static parsec_pins_callback * add_end_prev;
/*
static parsec_pins_callback * thread_init_prev; // courtesy calls to previously-registered cbs
static parsec_pins_callback * thread_fini_prev;
*/

static int 
    pins_prof_papi_L12_exec_begin, 
	pins_prof_papi_L12_exec_end, 
    pins_prof_papi_L12_select_begin, 
	pins_prof_papi_L12_select_end, 
    pins_prof_papi_L12_add_begin, 
	pins_prof_papi_L12_add_end, 
	pins_prof_papi_L123_begin, 
	pins_prof_papi_L123_end,
    pins_prof_papi_L123_starve_begin, 
	pins_prof_papi_L123_starve_end;

static void pins_init_papi_L123(dague_context_t * master_context) {
	(void)master_context;
	/*
    thread_init_prev = PINS_REGISTER(THREAD_INIT, start_papi_L123);
    thread_fini_prev = PINS_REGISTER(THREAD_FINI, stop_papi_L123);
	*/
	dague_profiling_add_dictionary_keyword("PINS_L123", "fill:#00AAFF",
	                                       sizeof(papi_L123_info_t), NULL,
	                                       &pins_prof_papi_L123_begin, 
										   &pins_prof_papi_L123_end);

	exec_begin_prev = PINS_REGISTER(EXEC_BEGIN, read_papi_L12_exec_count_begin);
	exec_end_prev   = PINS_REGISTER(EXEC_END, read_papi_L12_exec_count_end);
	dague_profiling_add_dictionary_keyword("PINS_L12_EXEC", "fill:#00FF00",
	                                       sizeof(papi_L12_exec_info_t), NULL,
	                                       &pins_prof_papi_L12_exec_begin, 
										   &pins_prof_papi_L12_exec_end);

	select_begin_prev = PINS_REGISTER(SELECT_BEGIN, read_papi_L12_select_count_begin);
	select_end_prev   = PINS_REGISTER(SELECT_END, read_papi_L12_select_count_end);
	dague_profiling_add_dictionary_keyword("PINS_L12_SELECT", "fill:#FFAA00",
	                                       sizeof(papi_L12_select_info_t), NULL,
	                                       &pins_prof_papi_L12_select_begin, 
										   &pins_prof_papi_L12_select_end);

	add_begin_prev = PINS_REGISTER(ADD_BEGIN, read_papi_L12_add_count_begin);
	add_end_prev   = PINS_REGISTER(ADD_END, read_papi_L12_add_count_end);
	dague_profiling_add_dictionary_keyword("PINS_L12_ADD", "fill:#AAFF00",
	                                       sizeof(papi_L12_exec_info_t), NULL,
	                                       &pins_prof_papi_L12_add_begin, 
										   &pins_prof_papi_L12_add_end);

	dague_profiling_add_dictionary_keyword("PINS_L123_STARVE", "fill:#FF8888",
										   sizeof(long long int), NULL,
										   &pins_prof_papi_L123_starve_begin,
										   &pins_prof_papi_L123_starve_end);
}

static void pins_fini_papi_L123(dague_context_t * master_context) {
	(void)master_context;
	// replace original registrants
	PINS_REGISTER(EXEC_BEGIN, exec_begin_prev);
	PINS_REGISTER(EXEC_END,   exec_end_prev);
	PINS_REGISTER(SELECT_BEGIN, select_begin_prev);
	PINS_REGISTER(SELECT_END,   select_end_prev);
	PINS_REGISTER(ADD_BEGIN, add_begin_prev);
	PINS_REGISTER(ADD_END,   add_end_prev);
	/*
	PINS_REGISTER(THREAD_INIT, thread_init_prev);
	PINS_REGISTER(THREAD_FINI, thread_fini_prev);
	*/
}

static void pins_thread_init_papi_L123(dague_execution_unit_t * exec_unit) {
	int rv = 0;
    int native;
	dague_profiling_trace(exec_unit->eu_profile,
						  pins_prof_papi_L123_starve_begin, 27, 0, NULL);

	exec_unit->papi_eventsets[EXEC_SET] = PAPI_NULL;
	if ((rv = PAPI_create_eventset(&exec_unit->papi_eventsets[EXEC_SET])) != PAPI_OK)
		printf("papi_L123_module.c, pins_thread_init_papi_L123: "
			   "thread %d couldn't create the PAPI event set "
			   "to measure L1/L2 misses; ERROR: %s\n", 
			   exec_unit->th_id, PAPI_strerror(rv));
	else {
		int i = 0;
		for (; i < NUM_L12_EVENTS; i++) {
			if (PAPI_event_name_to_code(L12_events[i], &native) != PAPI_OK)
				printf("papi_L123 couldn't find event %s.\n", L12_events[i]);
			else if (PAPI_add_event(exec_unit->papi_eventsets[EXEC_SET], native) 
					 != PAPI_OK)
				printf("papi_L123 couldn't add event %s.\n", L12_events[i]);
		}
		if (exec_unit->th_id % CORES_PER_SOCKET == WHICH_CORE_IN_SOCKET) {
			if (PAPI_event_name_to_code(L3_event, &native) != PAPI_OK)
				printf("papi_L123 couldn't find event %s.\n", L3_event);
			else if ((rv = PAPI_add_event(exec_unit->papi_eventsets[EXEC_SET], native))
					 != PAPI_OK)
				printf("papi_L123 couldn't add event %s, ERROR: %s\n", L3_event, 
					   PAPI_strerror(rv));
		}
		// start the event set (why wait?)
		if ((rv = PAPI_start(exec_unit->papi_eventsets[EXEC_SET])) != PAPI_OK)
			printf("papi_L123 couldn't start PAPI event set for thread "
				   "%d to measure L1, L2, and L3 misses; ERROR: %s\n", 
				   exec_unit->th_id, PAPI_strerror(rv));
		else {
			rv = dague_profiling_trace(exec_unit->eu_profile, 
									   pins_prof_papi_L123_begin, 
									   48, 0, NULL);
		}
	}
}

static void pins_thread_fini_papi_L123(dague_execution_unit_t * exec_unit) {
	long long int values[NUM_L12_EVENTS + 1];
	values[2] = 0;
	int rv = PAPI_OK;
	if ((rv = PAPI_stop(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
		printf("papi_L123_module.c, pins_thread_fini_papi_L123: thread %d couldn't stop "
			   "PAPI event set %d "
			   "to measure L3 misses; errno %d ERROR: %s\n",
			   exec_unit->th_id, exec_unit->papi_eventsets[EXEC_SET], 
			   errno,
			   PAPI_strerror(rv));
	}
	else {
		papi_L123_info_t info;
		info.vp_id = exec_unit->virtual_process->vp_id;
		info.th_id = exec_unit->th_id;
		info.L1_misses = values[0];
		info.L2_misses = values[1];
		info.L3_misses = values[2];
		rv = dague_profiling_trace(exec_unit->eu_profile, 
									pins_prof_papi_L123_end, 
									48, 0, (void *)&info);
	}

	dague_profiling_trace(exec_unit->eu_profile,
						  pins_prof_papi_L123_starve_end, 27, 0, 
						  (void *)&exec_unit->starvation);
}


static void read_papi_L12_exec_count_begin(dague_execution_unit_t * exec_unit, 
										   dague_execution_context_t * exec_context, 
										   void * data) {
	int rv = PAPI_OK;
	long long int values[NUM_L12_EVENTS + 1];
	values[2] = 0;
	if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
		printf("exec_begin: couldn't read PAPI events in thread %d\n", exec_unit->th_id);
	}
	else {
		rv = dague_profiling_trace(exec_unit->eu_profile, pins_prof_papi_L12_exec_begin, 
								   (*exec_context->function->key
									   )(exec_context->dague_handle, exec_context->locals), 
								   exec_context->dague_handle->handle_id, 
								   (void *)NULL);
		exec_unit->papi_last_read[0] = values[0];
		exec_unit->papi_last_read[1] = values[1];
		exec_unit->papi_last_read[2] = values[2];
	}
	// keep the contract with the previous registrant
	if (exec_begin_prev != NULL) {
		(*exec_begin_prev)(exec_unit, exec_context, data);
	}
}

static void read_papi_L12_exec_count_end(dague_execution_unit_t * exec_unit, 
										 dague_execution_context_t * exec_context, 
										 void * data) {
	int rv = PAPI_OK;
	long long int values[NUM_L12_EVENTS + 1];
	values[2] = 0;
	if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
		printf("exec_end: couldn't read PAPI events in thread %d\n", exec_unit->th_id);
	}
	else {
		papi_L12_exec_info_t info;
		info.kernel_type = exec_context->function->function_id;
		strncpy(info.kernel_name, exec_context->function->name, KERNEL_NAME_SIZE - 1);
		info.kernel_name[KERNEL_NAME_SIZE - 1] = '\0';
		info.vp_id = exec_unit->virtual_process->vp_id;
		info.th_id = exec_unit->th_id;
		info.L1_misses = values[0] - exec_unit->papi_last_read[0];
		info.L2_misses = values[1] - exec_unit->papi_last_read[1];
		info.L3_misses = values[2] - exec_unit->papi_last_read[2];

		rv = dague_profiling_trace(exec_unit->eu_profile, pins_prof_papi_L12_exec_end, 
								   (*exec_context->function->key
									   )(exec_context->dague_handle, exec_context->locals), 
								   exec_context->dague_handle->handle_id, 
								   (void *)&info);
		exec_unit->papi_last_read[0] = values[0];
		exec_unit->papi_last_read[1] = values[1];
		exec_unit->papi_last_read[2] = values[2];
	}
	// keep the contract with the previous registrant
	if (exec_end_prev != NULL) {
		(*exec_end_prev)(exec_unit, exec_context, data);
	}
}

static void read_papi_L12_select_count_begin(dague_execution_unit_t * exec_unit, 
										   dague_execution_context_t * exec_context, 
										   void * data) {
	int rv = PAPI_OK;
	long long int values[NUM_L12_EVENTS + 1];
	values[2] = 0;
	if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
		printf("select_begin: couldn't read PAPI events in thread %d, ERROR: %s\n", 
			   exec_unit->th_id, PAPI_strerror(rv));
	}
	else {
		rv = dague_profiling_trace(exec_unit->eu_profile, 
								   pins_prof_papi_L12_select_begin, 
								   32,
								   0,
								   (void *)NULL);
		exec_unit->papi_last_read[0] = values[0];
		exec_unit->papi_last_read[1] = values[1];
		exec_unit->papi_last_read[2] = values[2];
	}
	// keep the contract with the previous registrant
	if (select_begin_prev != NULL) {
		(*select_begin_prev)(exec_unit, exec_context, data);
	}
}

static void read_papi_L12_select_count_end(dague_execution_unit_t * exec_unit, 
										 dague_execution_context_t * exec_context, 
										 void * data) {
    unsigned long long victim_core_num = (unsigned long long)data;
    unsigned int num_threads = (exec_unit->virtual_process->dague_context->nb_vp 
                                * exec_unit->virtual_process->nb_cores);
	papi_L12_select_info_t info;
    if (exec_context) {
        info.kernel_type = exec_context->function->function_id;
		strncpy(info.kernel_name, exec_context->function->name, KERNEL_NAME_SIZE - 1);
		info.kernel_name[KERNEL_NAME_SIZE - 1] = '\0';
	}
    else {
        info.kernel_type = 0;
		strncpy(info.kernel_name, "<STARVED>", KERNEL_NAME_SIZE - 1);
		info.kernel_name[KERNEL_NAME_SIZE - 1] = '\0';
	}
    info.vp_id = exec_unit->virtual_process->vp_id;
    info.th_id = exec_unit->th_id;
    info.victim_vp_id = -1; // currently unavailable from scheduler queue object
    if (victim_core_num >= num_threads)
        info.victim_vp_id = SYSTEM_QUEUE_VP;
    info.victim_th_id = (int)victim_core_num; // but this number includes the vp id multiplier
    info.exec_context = (unsigned long long int)exec_context; // if NULL, this was starvation

    // now count the PAPI events, if available
	int rv = PAPI_OK;
	long long int values[NUM_L12_EVENTS + 1];
	values[2] = 0;
	if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
		printf("select_end: couldn't read PAPI events in thread %d, ERROR: %s\n", 
			   exec_unit->th_id, PAPI_strerror(rv));
		info.L1_misses = 0;
		info.L2_misses = 0;
		info.L3_misses = 0;
	}
	else {
		info.L1_misses = values[0] - exec_unit->papi_last_read[0];
		info.L2_misses = values[1] - exec_unit->papi_last_read[1];
		info.L3_misses = values[2] - exec_unit->papi_last_read[2];

		exec_unit->papi_last_read[0] = values[0];
		exec_unit->papi_last_read[1] = values[1];
		exec_unit->papi_last_read[2] = values[2];
	}

	rv = dague_profiling_trace(exec_unit->eu_profile,
							   pins_prof_papi_L12_select_end, 
							   32,
							   0,
							   (void *)&info);

	// keep the contract with the previous registrant
	if (select_end_prev != NULL) {
		(*select_end_prev)(exec_unit, exec_context, data);
	}
}

static void read_papi_L12_add_count_begin(dague_execution_unit_t * exec_unit, 
										   dague_execution_context_t * exec_context, 
										   void * data) {
	int rv = PAPI_OK;
	long long int values[NUM_L12_EVENTS + 1];
	values[2] = 0;
	if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
		printf("add_begin: couldn't read PAPI events in thread %d, ERROR: %s\n", 
			   exec_unit->th_id, PAPI_strerror(rv));
	}
	else {
		rv = dague_profiling_trace(exec_unit->eu_profile, 
								   pins_prof_papi_L12_add_begin, 
								   31,
								   0,
								   (void *)NULL);
		exec_unit->papi_last_read[0] = values[0];
		exec_unit->papi_last_read[1] = values[1];
		exec_unit->papi_last_read[2] = values[2];
	}
	// keep the contract with the previous registrant
	if (add_begin_prev != NULL) {
		(*add_begin_prev)(exec_unit, exec_context, data);
	}
}

static void read_papi_L12_add_count_end(dague_execution_unit_t * exec_unit, 
										 dague_execution_context_t * exec_context, 
										 void * data) {
	papi_L12_exec_info_t info;
	info.kernel_type = exec_context->function->function_id;
	strncpy(info.kernel_name, exec_context->function->name, KERNEL_NAME_SIZE - 1);
	info.kernel_name[KERNEL_NAME_SIZE - 1] = '\0';

    info.vp_id = exec_unit->virtual_process->vp_id;
    info.th_id = exec_unit->th_id;

    // now count the PAPI events, if available
	int rv = PAPI_OK;
	long long int values[NUM_L12_EVENTS + 1];
	values[2] = 0;
	if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
		printf("add_end: couldn't read PAPI events in thread %d, ERROR: %s\n", 
			   exec_unit->th_id, PAPI_strerror(rv));
		info.L1_misses = 0;
		info.L2_misses = 0;
		info.L3_misses = 0;
	}
	else {
		info.L1_misses = values[0] - exec_unit->papi_last_read[0];
		info.L2_misses = values[1] - exec_unit->papi_last_read[1];
		info.L3_misses = values[2] - exec_unit->papi_last_read[2];

		exec_unit->papi_last_read[0] = values[0];
		exec_unit->papi_last_read[1] = values[1];
		exec_unit->papi_last_read[2] = values[2];
	}

	rv = dague_profiling_trace(exec_unit->eu_profile,
							   pins_prof_papi_L12_add_end, 
							   31,
							   0,
							   (void *)&info);

	// keep the contract with the previous registrant
	if (add_end_prev != NULL) {
		(*add_end_prev)(exec_unit, exec_context, data);
	}
}

