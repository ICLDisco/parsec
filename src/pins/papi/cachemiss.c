#include "dague_config.h"
#include <papi.h>
#include "cachemiss.h"
#include <stdio.h>
#include "execution_unit.h"
#include "profiling.h"
#include "dague/pins/pins.h"
#include "shared_L3_misses.h"

static int pins_prof_exec_papi_core_begin, pins_prof_exec_papi_core_end;
static int exec_events[NUM_EXEC_EVENTS] = {PAPI_RES_STL, PAPI_L2_DCH, PAPI_L2_DCM, PAPI_L1_ICM};

void pins_init_cachemiss(dague_context_t * master_context) {
	(void)master_context;

	PINS_REGISTER(EXEC_BEGIN, start_papi_exec_count);
	PINS_REGISTER(EXEC_END, stop_papi_exec_count);
	// PETER TODO add requirement for DAGUE_PROF_TRACE
	dague_profiling_add_dictionary_keyword("PINS_EXEC_PAPI_CORE", "fill:#00FF00",
	                                       sizeof(pins_cachemiss_info_t), NULL,
	                                       &pins_prof_exec_papi_core_begin, &pins_prof_exec_papi_core_end);
}

void pins_thread_init_cachemiss(dague_execution_unit_t * exec_unit) {
	int rv = 0;
	if (exec_unit->th_id % CORES_PER_SOCKET != WHICH_CORE_FOR_L3 
	    || !DO_L3_MEASUREMENTS) {
		exec_unit->papi_eventsets[EXEC_SET] = PAPI_NULL;
		if (PAPI_create_eventset(&exec_unit->papi_eventsets[EXEC_SET]) != PAPI_OK)
			printf("cachemiss.c, pins_thread_init_cachemiss: failed to create ExecEventSet\n");
 		if ((rv = PAPI_add_events(exec_unit->papi_eventsets[EXEC_SET], exec_events, NUM_EXEC_EVENTS)) 
		    != PAPI_OK)
			printf("cachemiss.c, pins_thread_init_cachemiss: failed to add "
			       "exec events to ExecEventSet. %d %s\n", rv, PAPI_strerror(rv));
	}
}

void start_papi_exec_count(dague_execution_unit_t * exec_unit, 
                           dague_execution_context_t * exec_context, void * data) {
	(void)exec_context;
	(void)data;
	int rv = PAPI_OK;
	if (exec_unit->th_id % CORES_PER_SOCKET != WHICH_CORE_FOR_L3 
	    || !DO_L3_MEASUREMENTS) {
		if ((rv = PAPI_start(exec_unit->papi_eventsets[EXEC_SET])) != PAPI_OK) {
			printf("cachemiss.c, start_papi_exec_count: can't start "
			       "exec event counters! %d %s\n", 
			       rv, PAPI_strerror(rv));
		}
		else {
			dague_profiling_trace(exec_unit->eu_profile, pins_prof_exec_papi_core_begin, 
			                      (*exec_context->function->key
			                       )(exec_context->dague_handle, exec_context->locals), 
			                      exec_context->dague_handle->handle_id, NULL);
		}
	}		    
}

void stop_papi_exec_count(dague_execution_unit_t * exec_unit, 
                          dague_execution_context_t * exec_context, void * data) {
	(void)exec_context;
	(void)data;
	long long int values[NUM_EXEC_EVENTS];
	int rv = PAPI_OK;
	if (exec_unit->th_id % CORES_PER_SOCKET != WHICH_CORE_FOR_L3 
	    || !DO_L3_MEASUREMENTS) {
		if ((rv = PAPI_stop(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
			printf("cachemiss.c, stop_papi_exec_count: can't stop exec event counters! %d %s\n", 
			       rv, PAPI_strerror(rv));
		}
		else {
			pins_cachemiss_info_t info;
			info.kernel_type = exec_context->function->function_id;
			info.vp_id = exec_unit->virtual_process->vp_id;
			info.th_id = exec_unit->th_id;
			for(int i = 0; i < NUM_EXEC_EVENTS; i++) 
				info.values[i] = values[i];
			info.values_len = NUM_EXEC_EVENTS; /* not *necessary*, but perhaps better for compatibility
			                                    * with the Python dbpreader script in the long term,
			                                    * since this will allow the reading of different structs.
			                                    * presumably, a 'generic' Cython info reader could be created
			                                    * that allows a set of ints and a set of long longs
			                                    * to be automatically read if both lengths are included,
			                                    * e.g. struct { int num_ints; int; int; int; int num_lls;
			                                    * ll; ll; ll; ll; ll; ll } - the names could be assigned
			                                    * after the fact by a knowledgeable end user */
			dague_profiling_trace(exec_unit->eu_profile, pins_prof_exec_papi_core_end, 
			                      (*exec_context->function->key
			                       )(exec_context->dague_handle, exec_context->locals), 
			                      exec_context->dague_handle->handle_id, 
			                      (void *)&info);
		}
	}
}

void pins_handle_init_cachemiss(dague_handle_t * handle) {
	/*
	int rv;
	int num_cores = 0;
	unsigned int i;
	for (i = 0; i < handle->context->nb_vp; i++)
		num_cores += handle->context->virtual_processes[i]->nb_cores;
	
	for (i = 0; i < num_cores; i++) {
		handle->pins_data[i][EXEC_BEGIN] = calloc(sizeof(pins_cachemiss_data_t), 1);
		DATA(handle, i)->ExecEventSet = PAPI_NULL;
		if (PAPI_create_eventset(&(DATA(handle, i)->ExecEventSet)) != PAPI_OK)
			DEBUG(("cachemiss.c, pins_handle_init_cachemiss: failed to create ExecEventSet\n"));
       if ((rv = PAPI_add_events(DATA(handle, i)->ExecEventSet, exec_events, NUM_EXEC_EVENTS)) != PAPI_OK)
	       DEBUG(("cachemiss.c, pins_handle_init_cachemiss: failed to add exec events to ExecEventSet. %d %s\n", rv, PAPI_strerror(rv)));
       DATA(handle, i)->exec_cache_misses[0] = 0; // technically unnecessary b/c calloc
       DATA(handle, i)->exec_cache_misses[1] = 0;
       DATA(handle, i)->exec_tlb_misses = 0;

       DATA(handle, i)->StealEventSet = PAPI_NULL;
       if (PAPI_create_eventset(&DATA(handle, i)->StealEventSet) != PAPI_OK)
           printf("cachemiss.c, pins_handle_init_cachemiss: failed to create StealEventSet\n");
       if (PAPI_add_events(DATA(handle, i)->StealEventSet, steal_events, NUM_STEAL_EVENTS) != PAPI_OK)
           printf("cachemiss.c, pins_handle_init_cachemiss: failed to add steal events to StealEventSet\n");
        
	}	
	 */
}
