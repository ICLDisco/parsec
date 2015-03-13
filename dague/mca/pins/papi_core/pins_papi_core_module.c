#include "pins_papi_core.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "dague/utils/output.h"
#include "dague/utils/mca_param.h"
#include <stdio.h>
#include <papi.h>
#include "execution_unit.h"

static void start_papi_core(dague_execution_unit_t * exec_unit,
                              dague_execution_context_t * exec_context,
                              void * data);
static void stop_papi_core(dague_execution_unit_t * exec_unit,
                             dague_execution_context_t * exec_context,
                             void * data);

/* Courtesy calls to previously-registered cbs */
static parsec_pins_callback * exec_begin_prev;
static parsec_pins_callback * exec_end_prev;

static int pins_prof_papi_core_begin, pins_prof_papi_core_end;
 
static char* mca_param_string;

static void pins_init_papi_core(dague_context_t * master_context)
{
	pins_papi_init(master_context);
	
	dague_mca_param_reg_string_name("pins", "core_event",
	                                "PAPI event to be saved.\n",
	                                false, false,
	                                "", &mca_param_string);
    
    /* Handcrafted event to be added to the profiling */
    /* The first argument has to be "PINS_EXEC" for now. */
	dague_profiling_add_dictionary_keyword("PINS_CORE", "fill:#00AAFF",
	                                       sizeof(papi_core_info_t), NULL,
	                                       &pins_prof_papi_core_begin,
	                                       &pins_prof_papi_core_end);
    
    /* prepare link to the previously registered PINS module */
    exec_begin_prev = PINS_REGISTER(EXEC_BEGIN, start_papi_core);
    exec_end_prev = PINS_REGISTER(EXEC_END, stop_papi_core);
}

static void pins_fini_papi_core(dague_context_t * master_context)
{
    (void) master_context;
    
    PINS_REGISTER(EXEC_BEGIN, exec_begin_prev);
    PINS_REGISTER(EXEC_END, exec_end_prev);
}

static void pins_thread_init_papi_core(dague_execution_unit_t * exec_unit)
{
    char* mca_param_name;
    char* token;
    char* temp;
    int err, i;
    bool socket = false, core = false, started = false;
    
    exec_unit->num_core_counters = 0;
    exec_unit->pins_papi_core_event_name = (char**)calloc(NUM_CORE_EVENTS, sizeof(char*));
	exec_unit->pins_papi_core_native_event = (int*)calloc(NUM_CORE_EVENTS, sizeof(int));
	
	for(i = 0; i < NUM_CORE_EVENTS; i++)
	{
		exec_unit->pins_papi_core_event_name[i] = NULL;
		exec_unit->pins_papi_core_native_event[i] = PAPI_NULL;
	}
	
	asprintf(&mca_param_name, "%s", mca_param_string);

	token = strtok(mca_param_name, ":");
	
	if(token == NULL)
	{
		dague_output(0, "No PAPI events have been specified.  None will be recorded.\n");
		exec_unit->papi_eventsets[PER_CORE_SET] = PAPI_NULL;
		return;
	}
	
	while(token != NULL)
	{
		if(token[0] == 'S')
		{
            temp = (char*)calloc(strlen(token), sizeof(char));
            strcpy(temp, token);
            memmove(temp, temp+1, strlen(temp));
            
			if(temp[0] != '*')
			{
				if(atoi(temp) == exec_unit->socket_id)
					socket = true;
			}
			else
				socket = true;
            free(temp);
		}
		
		token = strtok(NULL, ":");
		
		if(token[0] == 'C')
		{
            temp = (char*)calloc(strlen(token),sizeof(char));
            strcpy(temp, token);
            memmove(temp, temp+1, strlen(temp));
            
			if(temp[0] != '*')
			{
				if(atoi(temp) == (exec_unit->core_id % CORES_PER_SOCKET))
					core = true;
			}
			else
				core = true;
            free(temp);
		}
		
		token = strtok(NULL, ",");
		
		if(socket && core)
		{
			if(exec_unit->num_core_counters == NUM_CORE_EVENTS)
			{
				dague_output(0, "pins_thread_init_papi_core: thread %d couldn't add event '%s' because only %d events are allowed.\n",
						exec_unit->th_id, token, NUM_CORE_EVENTS);
				break;
			}
			
			if(!started)
			{
				pins_papi_thread_init(exec_unit);
				exec_unit->papi_eventsets[PER_CORE_SET] = PAPI_NULL;
				
				/* Create an empty eventset */
				if( PAPI_OK != (err = PAPI_create_eventset(&exec_unit->papi_eventsets[PER_CORE_SET])) ) 
				{
					dague_output(0, "pins_thread_init_papi_core: thread %d couldn't create the PAPI event set; ERROR: %s\n",
						         exec_unit->th_id, PAPI_strerror(err));
					return;
				}
				started = true;
			}
			
			/* Convert event name to code */
			if(PAPI_OK == PAPI_event_name_to_code(token, &exec_unit->pins_papi_core_native_event[exec_unit->num_core_counters]) )
			{
				exec_unit->pins_papi_core_event_name[exec_unit->num_core_counters] = (char*)calloc(strlen(token), sizeof(char));
				strcpy(exec_unit->pins_papi_core_event_name[exec_unit->num_core_counters], token);
			}
			
			if(PAPI_NULL == exec_unit->pins_papi_core_native_event[exec_unit->num_core_counters])
			{
				dague_output(0, "No event derived from %s is supported on this system (use papi_native_avail for a complete list)\n",
		                 token);
		   		return;
			}
		
			/* Add events to the eventset */
			if( PAPI_OK != (err = PAPI_add_event(exec_unit->papi_eventsets[PER_CORE_SET],
				                                 exec_unit->pins_papi_core_native_event[exec_unit->num_core_counters])) ) 
			{
				dague_output(0, "pins_thread_init_papi_core: failed to add event %s; ERROR: %s\n",
				             token, PAPI_strerror(err));
				return;
			}
			exec_unit->num_core_counters++;
		}
		
		socket = false;
		core = false;
		token = strtok(NULL, ":");
	}
	
	free(mca_param_name);
	free(token);
    
    if(exec_unit->num_core_counters > 0)
    {
		/* Start the PAPI counters. */
		if( PAPI_OK != (err = PAPI_start(exec_unit->papi_eventsets[PER_CORE_SET])) ) 
		{
		    dague_output(0, "couldn't start PAPI eventset for thread %d; ERROR: %s\n",
		                 exec_unit->th_id, PAPI_strerror(err));
		}
    }
    else
    {
    	exec_unit->papi_eventsets[PER_CORE_SET] = PAPI_NULL;
    }
}

static void pins_thread_fini_papi_core(dague_execution_unit_t * exec_unit)
{
    int err, i;
    long long int values[NUM_CORE_EVENTS];
	
	if( PAPI_NULL != exec_unit->papi_eventsets[PER_CORE_SET] )
	{
		if( PAPI_OK != (err = PAPI_read(exec_unit->papi_eventsets[PER_CORE_SET], values)) ) 
		{
			dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
			             exec_unit->th_id, PAPI_strerror(err));
		}

		int inc = 0;
		papi_core_info_t info;
		/*info.vp_id = exec_unit->virtual_process->vp_id;*/
		info.th_id = exec_unit->th_id;
	
		info.kernel_type = -1;
		/* exec_context isn't available here.
		if (exec_context->dague_handle->profiling_array != NULL)
		        info.kernel_type = exec_context->dague_handle->profiling_array[exec_context->function->function_id * 2] / 2;
 		*/
	
		for(int i = 0; i < NUM_CORE_EVENTS; i++)
		{
			info.values[i] = values[i];
		}
	
		info.values_len = NUM_CORE_EVENTS;
		
		inc = dague_profiling_trace(exec_unit->eu_profile,
			                        pins_prof_papi_core_end, 55, 0, (void *)&info);
		inc = dague_profiling_trace(exec_unit->eu_profile,
			                        pins_prof_papi_core_end, 55, 0, (void *)&info);
	}
	
    for(i = 0; i < NUM_CORE_EVENTS; i++)
    {
		if( PAPI_NULL == exec_unit->pins_papi_core_native_event[i] )
		    return;  /* nothing to see here */
    }
	
	/* Stop the PAPI counters. */
    if( PAPI_OK != (err = PAPI_stop(exec_unit->papi_eventsets[PER_CORE_SET], values)) ) 
    {
        dague_output(0, "couldn't stop PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
    }
	
    /* the counting should be stopped by now */
    for(i = 0; i < NUM_CORE_EVENTS; i++)
    {
		if( PAPI_OK != (err = PAPI_remove_event(exec_unit->papi_eventsets[PER_CORE_SET],
		                                        exec_unit->pins_papi_core_native_event[i])) ) 
		{
		    dague_output(0, "pins_thread_fini_papi_core: failed to remove event %s; ERROR: %s\n",
		                 exec_unit->pins_papi_core_event_name[i], PAPI_strerror(err));
		}
    }
    
    for(i = 0; i < NUM_CORE_EVENTS; i++)
    {
    	free(exec_unit->pins_papi_core_event_name[i]);
    }
    
    free(exec_unit->pins_papi_core_event_name);
    free(exec_unit->pins_papi_core_native_event);
    
    if( PAPI_OK != (err = PAPI_cleanup_eventset(exec_unit->papi_eventsets[PER_CORE_SET])) ) {
        dague_output(0, "pins_thread_fini_papi_core: failed to cleanup thread %d eventset; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
    }
    if( PAPI_OK != (err = PAPI_destroy_eventset(&exec_unit->papi_eventsets[PER_CORE_SET])) ) {
        dague_output(0, "pins_thread_fini_papi_core: failed to destroy thread %d eventset; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
    }
    
}

static void start_papi_core(dague_execution_unit_t * exec_unit,
                              dague_execution_context_t * exec_context,
                              void * data)
{
    int err = 0;

    if( PAPI_NULL == exec_unit->papi_eventsets[PER_CORE_SET] )
        goto next_pins;
   
	long long int values[NUM_CORE_EVENTS];
	
	if( PAPI_OK != (err = PAPI_read(exec_unit->papi_eventsets[PER_CORE_SET], values)) ) 
	{
	    dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
	                 exec_unit->th_id, PAPI_strerror(err));
	    goto next_pins;
	}
	
	int inc = 0;
	papi_core_info_t info;
	/*info.vp_id = exec_unit->virtual_process->vp_id;*/
	info.th_id = exec_unit->th_id;
	
	info.kernel_type = -1;
	if (exec_context->dague_handle->profiling_array != NULL)
            info.kernel_type = exec_context->dague_handle->profiling_array[exec_context->function->function_id * 2] / 2;
	
	for(int i = 0; i < NUM_CORE_EVENTS; i++)
	{
	    info.values[i] = values[i];
	}
	
	info.values_len = NUM_CORE_EVENTS;
	
	inc = dague_profiling_trace(exec_unit->eu_profile,
	                            pins_prof_papi_core_begin, 55, 0, (void *)&info);
	
  next_pins:
    /* call previous callback, if any */
    if (NULL != exec_begin_prev) {
        (*exec_begin_prev)(exec_unit, exec_context, data);
    }
    (void)exec_context; (void)data;
}

static void stop_papi_core(dague_execution_unit_t * exec_unit,
                             dague_execution_context_t * exec_context,
                             void * data)
{
    if( PAPI_NULL == exec_unit->papi_eventsets[PER_CORE_SET] )
        goto next_pins;
	
	long long int values[NUM_CORE_EVENTS];
	int err;
	
	if( PAPI_OK != (err = PAPI_read(exec_unit->papi_eventsets[PER_CORE_SET], values)) ) 
	{
	    dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
	                 exec_unit->th_id, PAPI_strerror(err));
	    goto next_pins;
	}

	int inc = 0;
	papi_core_info_t info;
	/*info.vp_id = exec_unit->virtual_process->vp_id;*/
	info.th_id = exec_unit->th_id;
	
	info.kernel_type = -1;
	if (exec_context->dague_handle->profiling_array != NULL)
            info.kernel_type = exec_context->dague_handle->profiling_array[exec_context->function->function_id * 2] / 2;
	
	for(int i = 0; i < NUM_CORE_EVENTS; i++)
	{
	    info.values[i] = values[i];
	}
	
	info.values_len = NUM_CORE_EVENTS;
	
	inc = dague_profiling_trace(exec_unit->eu_profile,
	                            pins_prof_papi_core_end, 55, 0, (void *)&info);
	

  next_pins:
    /* call previous callback, if any */
    if (NULL != exec_end_prev)
        (*exec_end_prev)(exec_unit, exec_context, data);
    
    (void)exec_context; (void)data;
}

const dague_pins_module_t dague_pins_papi_core_module = {
    &dague_pins_papi_core_component,
    {
        pins_init_papi_core,
        pins_fini_papi_core,
        NULL,
        NULL,
        pins_thread_init_papi_core,
        pins_thread_fini_papi_core,
    }
};
