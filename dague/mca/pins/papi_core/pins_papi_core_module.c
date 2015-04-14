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

static char* mca_param_string;

static void pins_init_papi_core(dague_context_t * master_context) {
    pins_papi_init(master_context);

    dague_mca_param_reg_string_name("pins", "core_event",
                                    "PAPI event to be saved.\n",
                                    false, false,
                                    "", &mca_param_string);

    /* prepare link to the previously registered PINS module */
    exec_begin_prev = PINS_REGISTER(EXEC_BEGIN, start_papi_core);
    exec_end_prev = PINS_REGISTER(EXEC_END, stop_papi_core);
}

static void pins_fini_papi_core(dague_context_t * master_context) {
    (void) master_context;
    PINS_REGISTER(EXEC_BEGIN, exec_begin_prev);
    PINS_REGISTER(EXEC_END, exec_end_prev);
}

static void pins_thread_init_papi_core(dague_execution_unit_t * exec_unit) 
{
    int err, i;
    
    exec_unit->papi_eventsets[PER_CORE_SET] = PAPI_NULL;

    exec_unit->num_core_counters = pins_papi_mca_string_parse(exec_unit, mca_param_string, &exec_unit->pins_papi_core_event_names);
    if(exec_unit->num_core_counters == 0)
        return;

    if(-1 == pins_papi_create_eventset(exec_unit, &exec_unit->papi_eventsets[PER_CORE_SET], exec_unit->pins_papi_core_event_names,
                                       &exec_unit->pins_papi_core_native_events, exec_unit->num_core_counters)) {
        exec_unit->papi_eventsets[PER_CORE_SET] = PAPI_NULL;
        return;
    }
    
    exec_unit->core_values = (long long*)malloc(exec_unit->num_core_counters * sizeof(long long));
    
    /* Add the dictionary keyword and start the PAPI counters */
    char* key_string;
    char* value_string;
    int string_size = 0;

    asprintf(&key_string, "PINS_CORE_S%d_C%d", exec_unit->socket_id, exec_unit->core_id);

    for(i = 0; i < exec_unit->num_core_counters; i++) {
        string_size += strlen(exec_unit->pins_papi_core_event_names[i]) + strlen("{int64_t}"PARSEC_PAPI_SEPARATOR);
    }

    value_string = (char*)calloc(string_size, sizeof(char));

    for(i = 0; i < exec_unit->num_core_counters; i++) {
        strcat(value_string, exec_unit->pins_papi_core_event_names[i]);
        strcat(value_string, "{int64_t}"PARSEC_PAPI_SEPARATOR);
    }

    dague_profiling_add_dictionary_keyword(key_string, "fill:#00AAFF",
                                           sizeof(uint64_t) * exec_unit->num_core_counters, value_string,
                                           &exec_unit->pins_prof_papi_core[0],
                                           &exec_unit->pins_prof_papi_core[1]);
    free(key_string);
    free(value_string);

    if( PAPI_OK != (err = PAPI_start(exec_unit->papi_eventsets[PER_CORE_SET])) ) {
        dague_output(0, "couldn't start PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
    }
}

static void pins_thread_fini_papi_core(dague_execution_unit_t * exec_unit) {
    int err, i;

    if( PAPI_NULL == exec_unit->papi_eventsets[PER_CORE_SET] )
        return;

    if( PAPI_OK != (err = PAPI_stop(exec_unit->papi_eventsets[PER_CORE_SET], exec_unit->core_values)) ) {
        dague_output(0, "couldn't stop PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
    }
    /* the counting should be stopped by now */
    for(i = 0; i < exec_unit->num_core_counters; i++) {
        if( PAPI_OK != (err = PAPI_remove_event(exec_unit->papi_eventsets[PER_CORE_SET],
                                                exec_unit->pins_papi_core_native_events[i])) ) {
            dague_output(0, "pins_thread_fini_papi_core: failed to remove event %s; ERROR: %s\n",
                         exec_unit->pins_papi_core_event_names[i], PAPI_strerror(err));
        }
    }

    for(i = 0; i < exec_unit->num_core_counters; i++)
        free(exec_unit->pins_papi_core_event_names[i]);

    free(exec_unit->pins_papi_core_event_names);
    free(exec_unit->pins_papi_core_native_events);
    free(exec_unit->core_values);

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
    if( PAPI_NULL == exec_unit->papi_eventsets[PER_CORE_SET] )
        goto next_pins;

    int err;

    if( PAPI_OK != (err = PAPI_read(exec_unit->papi_eventsets[PER_CORE_SET], exec_unit->core_values)) ) {
        dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        goto next_pins;
    }

    (void)dague_profiling_trace(exec_unit->eu_profile, exec_unit->pins_prof_papi_core[0],
                                45, 0, (void *)exec_unit->core_values);

  next_pins:
    /* call previous callback, if any */
    if (NULL != exec_begin_prev)
        (*exec_begin_prev)(exec_unit, exec_context, data);

    (void)exec_context; (void)data;
}

static void stop_papi_core(dague_execution_unit_t * exec_unit,
                           dague_execution_context_t * exec_context,
                           void * data) {
    if( PAPI_NULL == exec_unit->papi_eventsets[PER_CORE_SET] )
        goto next_pins;

    int err;

    if( PAPI_OK != (err = PAPI_read(exec_unit->papi_eventsets[PER_CORE_SET], exec_unit->core_values)) ) {
        dague_output(0, "couldn't read PAPI eventset for thread %d; ERROR: %s\n",
                     exec_unit->th_id, PAPI_strerror(err));
        goto next_pins;
    }

    (void)dague_profiling_trace(exec_unit->eu_profile, exec_unit->pins_prof_papi_core[1],
                                45, 0, (void *)exec_unit->core_values);

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
