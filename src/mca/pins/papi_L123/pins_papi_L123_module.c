#include <errno.h>
#include "dague_config.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "pins_papi_L123.h"
#include <papi.h>
#include <stdio.h>
#include "profiling.h"
#include "execution_unit.h"
#ifdef PARSEC_PROF_TAU
#include "TAU.h"
#endif

static char* core_native_events   [NUM_CORE_EVENTS]   = PAPI_CORE_NATIVE_EVENT_ARRAY;
static char* socket_native_events [NUM_SOCKET_EVENTS] = PAPI_SOCKET_NATIVE_EVENT_ARRAY;

static int enable_socket = ENABLE_SOCKET; /* TODO: use MCA for these config */
static int enable_prep = ENABLE_PREP;
static int enable_select = ENABLE_SELECT;
static int enable_exec = ENABLE_EXEC;
static int enable_compl = ENABLE_COMPL;

static void pins_init_papi_L123(dague_context_t * master_context);
static void pins_fini_papi_L123(dague_context_t * master_context);
static void pins_thread_init_papi_L123(dague_execution_unit_t * exec_unit);
static void pins_thread_fini_papi_L123(dague_execution_unit_t * exec_unit);

static void read_papi_core_prep_count_begin(
    dague_execution_unit_t * exec_unit,
    dague_execution_context_t * exec_context,
    void * data);
static void read_papi_core_prep_count_end(
    dague_execution_unit_t * exec_unit,
    dague_execution_context_t * exec_context,
    void * data);
static void read_papi_core_select_count_begin(
    dague_execution_unit_t * exec_unit,
    dague_execution_context_t * exec_context,
    void * data);
static void read_papi_core_select_count_end(
    dague_execution_unit_t * exec_unit,
    dague_execution_context_t * exec_context,
    void * data);
static void read_papi_core_exec_count_begin(
    dague_execution_unit_t * exec_unit,
    dague_execution_context_t * exec_context,
    void * data);
static void read_papi_core_exec_count_end(
    dague_execution_unit_t * exec_unit,
    dague_execution_context_t * exec_context,
    void * data);
static void read_papi_core_complete_exec_count_begin(
    dague_execution_unit_t * exec_unit,
    dague_execution_context_t * exec_context,
    void * data);
static void read_papi_core_complete_exec_count_end(
    dague_execution_unit_t * exec_unit,
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

static parsec_pins_callback * prep_begin_prev = NULL; // courtesy calls to previously-registered cbs
static parsec_pins_callback * prep_end_prev = NULL;
static parsec_pins_callback * select_begin_prev = NULL; // courtesy calls to previously-registered cbs
static parsec_pins_callback * select_end_prev = NULL;
static parsec_pins_callback * exec_begin_prev = NULL; // courtesy calls to previously-registered cbs
static parsec_pins_callback * exec_end_prev = NULL;
static parsec_pins_callback * compl_begin_prev = NULL; // courtesy calls to previously-registered cbs
static parsec_pins_callback * compl_end_prev = NULL;

static int
    pins_prof_papi_socket_begin,
    pins_prof_papi_socket_end,
    pins_prof_papi_core_prep_begin,
    pins_prof_papi_core_prep_end,
    pins_prof_papi_core_select_begin,
    pins_prof_papi_core_select_end,
    pins_prof_papi_core_exec_begin,
    pins_prof_papi_core_exec_end,
    pins_prof_papi_core_compl_begin,
    pins_prof_papi_core_compl_end;

static void pins_init_papi_L123(dague_context_t * master_context) {
    pins_papi_init(master_context);
#ifdef PARSEC_PROF_TAU
    pins_tau_init(master_context);
#endif

    if (enable_socket)
        dague_profiling_add_dictionary_keyword(PAPI_CORE_PROF_EVT_NAME_SOCKET, "fill:#00AAFF",
                                               sizeof(papi_core_socket_info_t), NULL,
                                               &pins_prof_papi_socket_begin,
                                               &pins_prof_papi_socket_end);

    if (enable_prep) {
        prep_begin_prev = PINS_REGISTER(PREPARE_INPUT_BEGIN, read_papi_core_prep_count_begin);
        prep_end_prev   = PINS_REGISTER(PREPARE_INPUT_END, read_papi_core_prep_count_end);
        dague_profiling_add_dictionary_keyword(PAPI_CORE_PROF_EVT_NAME_PREP, "fill:#00FF00",
                                               sizeof(papi_core_exec_info_t), NULL,
                                               &pins_prof_papi_core_prep_begin,
                                               &pins_prof_papi_core_prep_end);
    }
    if (enable_select) {
        select_begin_prev = PINS_REGISTER(SELECT_BEGIN, read_papi_core_select_count_begin);
        select_end_prev   = PINS_REGISTER(SELECT_END, read_papi_core_select_count_end);
        dague_profiling_add_dictionary_keyword(PAPI_CORE_PROF_EVT_NAME_SEL, "fill:#FFAA00",
                                               sizeof(papi_core_select_info_t), NULL,
                                               &pins_prof_papi_core_select_begin,
                                               &pins_prof_papi_core_select_end);
    }
    if (enable_exec) {
        exec_begin_prev = PINS_REGISTER(EXEC_BEGIN, read_papi_core_exec_count_begin);
        exec_end_prev   = PINS_REGISTER(EXEC_END, read_papi_core_exec_count_end);
        dague_profiling_add_dictionary_keyword(PAPI_CORE_PROF_EVT_NAME_EXEC, "fill:#00FF00",
                                               sizeof(papi_core_exec_info_t), NULL,
                                               &pins_prof_papi_core_exec_begin,
                                               &pins_prof_papi_core_exec_end);
    }
    if (enable_compl) {
        compl_begin_prev = PINS_REGISTER(COMPLETE_EXEC_BEGIN, read_papi_core_complete_exec_count_begin);
        compl_end_prev   = PINS_REGISTER(COMPLETE_EXEC_END, read_papi_core_complete_exec_count_end);
        dague_profiling_add_dictionary_keyword(PAPI_CORE_PROF_EVT_NAME_COMPL, "fill:#AAFF00",
                                               sizeof(papi_core_exec_info_t), NULL,
                                               &pins_prof_papi_core_compl_begin,
                                               &pins_prof_papi_core_compl_end);
    }
}

static void pins_fini_papi_L123(dague_context_t * master_context) {
    (void)master_context;
    /* replace original registrants */
    if (enable_prep) {
        PINS_REGISTER(PREPARE_INPUT_BEGIN, prep_begin_prev);
        PINS_REGISTER(PREPARE_INPUT_END,   prep_end_prev);
    }
    if (enable_select) {
        PINS_REGISTER(SELECT_BEGIN,        select_begin_prev);
        PINS_REGISTER(SELECT_END,          select_end_prev);
    }
    if (enable_exec) {
        PINS_REGISTER(EXEC_BEGIN,          exec_begin_prev);
        PINS_REGISTER(EXEC_END,            exec_end_prev);
    }
    if (enable_compl) {
        PINS_REGISTER(COMPLETE_EXEC_BEGIN, compl_begin_prev);
        PINS_REGISTER(COMPLETE_EXEC_END,   compl_end_prev);
    }
}

static void pins_thread_init_papi_L123(dague_execution_unit_t * exec_unit) {
    int rv = 0;
    int native;

    pins_papi_thread_init(exec_unit);
#ifdef PARSEC_PROF_TAU
    pins_tau_thread_init(master_context);
#endif

    /* all threads can store their own start time */
    PROFILING_THREAD_SAVE_uint64INFO(exec_unit->eu_profile, "begin", dague_profiling_get_time());

    exec_unit->papi_eventsets[EXEC_SET] = PAPI_NULL;

    rv = PAPI_create_eventset(&exec_unit->papi_eventsets[EXEC_SET]);
    if (PAPI_OK != rv)
        fprintf(stderr,"pins_papi_L123_module.c, pins_thread_init_papi_L123: "
               "thread %d couldn't create the PAPI event set "
               "to measure PAPI events; ERROR: %s\n",
               exec_unit->th_id, PAPI_strerror(rv));
    else {
        int i = 0;
        for (; i < NUM_CORE_EVENTS; i++) {
            if (PAPI_OK != PAPI_event_name_to_code(
                    core_native_events[i], &native)) {
                /* error */
                fprintf(stderr,"papi_L123 thread %d couldn't find event %s.\n",
                       exec_unit->th_id,
                       core_native_events[i]);
            }
            else if (PAPI_OK != PAPI_add_event(
                         exec_unit->papi_eventsets[EXEC_SET], native)) {
                /* error */
                fprintf(stderr,"papi_L123 thread %d couldn't add event %s.\n",
                       exec_unit->th_id,
                       core_native_events[i]);
            }
            else {
                /* success */
                /* fprintf(stderr,"papi_L123 thread %d added event %s\n", */
                /*        exec_unit->th_id, */
                /*        core_native_events[i]); */
            }
        }
        if (enable_socket &&
            exec_unit->th_id % CORES_PER_SOCKET == WHICH_CORE_IN_SOCKET) {
            for (i = 0; i < NUM_SOCKET_EVENTS; i++) {
                if ( PAPI_OK != PAPI_event_name_to_code(
                         socket_native_events[i], &native)) {
                    /* error */
                    fprintf(stderr,"papi_L123 thread %d couldn't find event %s.\n",
                           exec_unit->th_id,
                           socket_native_events[i]);
                }
                else if (PAPI_OK != (rv = PAPI_add_event(
                                         exec_unit->papi_eventsets[EXEC_SET], native))) {
                    /* error */
                    fprintf(stderr,"papi_L123 thread %d couldn't add event %s, ERROR: %s\n",
                           exec_unit->th_id,
                           socket_native_events[i],
                           PAPI_strerror(rv));
                }
                else {
                    /* success */
                    /* fprintf(stderr,"papi_L123 thread %d added event %s\n", */
                    /*        exec_unit->th_id, */
                    /*        socket_native_events[i]); */
                }
            }
        }

        // start the event set (why wait?)
        rv = PAPI_start(exec_unit->papi_eventsets[EXEC_SET]);
        if (PAPI_OK != rv)
            fprintf(stderr,"papi_L123 couldn't start PAPI event set for thread %d; ERROR: %s\n",
                   exec_unit->th_id, PAPI_strerror(rv));
        else {
            if (enable_socket)
                rv = dague_profiling_trace(exec_unit->eu_profile,
                                           pins_prof_papi_socket_begin,
                                           48, 0, NULL);
        }
    }
}

static void pins_thread_fini_papi_L123(dague_execution_unit_t * exec_unit) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    errno = 0;

    rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values);
    if (PAPI_OK != rv) {
        fprintf(stderr,"papi_L123_module.c, pins_thread_fini_papi_L123: thread %d couldn't stop "
               "PAPI event set %d errno %d %s ERROR: %s\n",
               exec_unit->th_id, exec_unit->papi_eventsets[EXEC_SET],
               errno, strerror(errno),
               PAPI_strerror(rv));
    }
    else {
        papi_core_socket_info_t info;
        for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
            info.evt_values[rv] = values[rv];
        if (enable_socket)
            rv = dague_profiling_trace(exec_unit->eu_profile,
                                       pins_prof_papi_socket_end,
                                       48, 0, (void *)&info);
    }

    /* add thread 'end' info before dumping */
    PROFILING_THREAD_SAVE_uint64INFO(exec_unit->eu_profile, "end", dague_profiling_get_time());
}

static void read_papi_core_counters_and_trace(dague_execution_unit_t * exec_unit,
                                              dague_execution_context_t * exec_context,
                                              void * data, int trace_evt) {
    // do nothing...yet
}

static void read_papi_core_exec_count_begin(dague_execution_unit_t * exec_unit,
                                            dague_execution_context_t * exec_context,
                                            void * data) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values);
    if (PAPI_OK != rv) {
        fprintf(stderr, "exec_begin: couldn't read PAPI events in thread %d\n%d %s",
                exec_unit->th_id, rv, PAPI_strerror(rv));
    }
    else {
        rv = dague_profiling_trace(exec_unit->eu_profile, pins_prof_papi_core_exec_begin,
                                   (*exec_context->function->key
                                    )(exec_context->dague_handle, exec_context->locals),
                                   exec_context->dague_handle->handle_id,
                                   (void *)NULL);
        for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
            exec_unit->papi_last_read[rv] = values[rv];
    }
    // keep the contract with the previous registrant
    if (exec_begin_prev != NULL) {
        (*exec_begin_prev)(exec_unit, exec_context, data);
    }

#ifdef PARSEC_PROF_TAU
    TAU_START("exec");
#endif
}

static void read_papi_core_exec_count_end(dague_execution_unit_t * exec_unit,
                                         dague_execution_context_t * exec_context,
                                         void * data) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values);
    if (PAPI_OK != rv) {
        fprintf(stderr, "exec_end: couldn't read PAPI events in thread %d\n%d %s",
                exec_unit->th_id, rv, PAPI_strerror(rv));
    }
    else {
        papi_core_exec_info_t info;
        info.kernel_type = -1;
        if (exec_context->dague_handle->profiling_array != NULL)
            info.kernel_type = exec_context->dague_handle->profiling_array[exec_context->function->function_id * 2] / 2;
        for (rv = 0; rv < NUM_CORE_EVENTS; rv++) {
            info.evt_values[rv] = values[rv] - exec_unit->papi_last_read[rv];
        }
        rv = dague_profiling_trace(
            exec_unit->eu_profile, pins_prof_papi_core_exec_end,
            (*exec_context->function->key)(exec_context->dague_handle, exec_context->locals),
            exec_context->dague_handle->handle_id, (void *)&info);
        for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
            exec_unit->papi_last_read[rv] = values[rv];
    }
    // keep the contract with the previous registrant
    if (exec_end_prev != NULL) {
        (*exec_end_prev)(exec_unit, exec_context, data);
    }

#ifdef PARSEC_PROF_TAU
    TAU_STOP("exec");
#endif
}

static void read_papi_core_prep_count_begin(dague_execution_unit_t * exec_unit,
                                           dague_execution_context_t * exec_context,
                                           void * data) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values);
    if (PAPI_OK != rv) {
        fprintf(stderr, "prep_begin: couldn't read PAPI events in thread %d\n%d %s",
                exec_unit->th_id, rv, PAPI_strerror(rv));
    }
    else {
        rv = dague_profiling_trace(exec_unit->eu_profile, pins_prof_papi_core_prep_begin,
                                   (*exec_context->function->key
                                    )(exec_context->dague_handle, exec_context->locals),
                                   exec_context->dague_handle->handle_id,
                                   (void *)NULL);
        for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
            exec_unit->papi_last_read[rv] = values[rv];
    }
    // keep the contract with the previous registrant
    if (prep_begin_prev != NULL) {
        (*prep_begin_prev)(exec_unit, exec_context, data);
    }

#ifdef PARSEC_PROF_TAU
    TAU_START("prep");
#endif
}

static void read_papi_core_prep_count_end(dague_execution_unit_t * exec_unit,
                                         dague_execution_context_t * exec_context,
                                         void * data) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values);
    if (PAPI_OK != rv) {
        fprintf(stderr, "prep_end: couldn't read PAPI events in thread %d\n%d %s",
                exec_unit->th_id, rv, PAPI_strerror(rv));
    }
    else {
        papi_core_exec_info_t info;
        info.kernel_type = -1;
        if (exec_context->dague_handle->profiling_array != NULL)
            info.kernel_type = exec_context->dague_handle->profiling_array[exec_context->function->function_id * 2] / 2;
        for (rv = 0; rv < NUM_CORE_EVENTS; rv++)
            info.evt_values[rv] = values[rv] - exec_unit->papi_last_read[rv];
        rv = dague_profiling_trace(exec_unit->eu_profile, pins_prof_papi_core_prep_end,
                                   (*exec_context->function->key
                                    )(exec_context->dague_handle, exec_context->locals),
                                   exec_context->dague_handle->handle_id,
                                   (void *)&info);
        for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
            exec_unit->papi_last_read[rv] = values[rv];
    }
    // keep the contract with the previous registrant
    if (prep_end_prev != NULL) {
        (*prep_end_prev)(exec_unit, exec_context, data);
    }

#ifdef PARSEC_PROF_TAU
    TAU_STOP("prep");
#endif
}



static void read_papi_core_select_count_begin(dague_execution_unit_t * exec_unit,
                                              dague_execution_context_t * exec_context,
                                              void * data) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];

    rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values);
    if (PAPI_OK != rv) {
        fprintf(stderr, "select_begin: couldn't read PAPI events in thread %d, ERROR: %s\n",
               exec_unit->th_id, PAPI_strerror(rv));
    }
    else {
        rv = dague_profiling_trace(exec_unit->eu_profile,
                                   pins_prof_papi_core_select_begin,
                                   320,
                                   0,
                                   (void *)NULL);
        for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
            exec_unit->papi_last_read[rv] = values[rv];
    }
    /* keep the contract with the previous registrant */
    if (select_begin_prev != NULL) {
        (*select_begin_prev)(exec_unit, exec_context, data);
    }
}

static void read_papi_core_select_count_end(dague_execution_unit_t * exec_unit,
                                            dague_execution_context_t * exec_context,
                                            void * data) {
    if (exec_context) {
        unsigned long long victim_core_num = -1;
        unsigned int num_threads = (exec_unit->virtual_process->dague_context->nb_vp
                                    * exec_unit->virtual_process->nb_cores);
        papi_core_select_info_t info;

        info.kernel_type = -1;
        info.selection_time = (unsigned long long int)data;
        info.exec_context = (unsigned long long int)exec_context;
        info.victim_vp_id = -1; // currently unavailable from scheduler queue object

        /* is this a function with an identifier? */
        if (exec_context->dague_handle->profiling_array != NULL)
            info.kernel_type = exec_context->dague_handle->profiling_array[exec_context->function->function_id * 2] / 2;

        victim_core_num = exec_context->victim_core;
        if (victim_core_num >= num_threads)
            info.victim_vp_id = SYSTEM_QUEUE_VP;
        info.victim_th_id = (int)victim_core_num; /* this number includes the vp id multiplier */

        int rv = PAPI_OK;
        long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
        /* now count the PAPI events, if available */
        rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values);
        if (PAPI_OK != rv) {
            fprintf(stderr, "select_end: couldn't read PAPI events in thread %d, ERROR: %s\n",
                    exec_unit->th_id, PAPI_strerror(rv));
            for (rv = 0; rv < NUM_CORE_EVENTS; rv++)
                info.evt_values[rv] = 0;
        }
        else {
            for (rv = 0; rv < NUM_CORE_EVENTS; rv++)
                info.evt_values[rv] = values[rv] - exec_unit->papi_last_read[rv];
            for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
                exec_unit->papi_last_read[rv] = values[rv];
        }

        rv = dague_profiling_trace(exec_unit->eu_profile,
                                   pins_prof_papi_core_select_end,
                                   320,
                                   0,
                                   (void *)&info);
    }

    /* keep the contract with the previous registrant */
    if (select_end_prev != NULL) {
        (*select_end_prev)(exec_unit, exec_context, data);
    }
}

static void read_papi_core_complete_exec_count_begin(dague_execution_unit_t * exec_unit,
                                                    dague_execution_context_t * exec_context,
                                                    void * data) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values);
    if (PAPI_OK != rv) {
        fprintf(stderr, "compl_begin: couldn't read PAPI events in thread %d, ERROR: %s\n",
               exec_unit->th_id, PAPI_strerror(rv));
    }
    else {
        rv = dague_profiling_trace(exec_unit->eu_profile,
                                   pins_prof_papi_core_compl_begin,
                                   31,
                                   0,
                                   (void *)NULL);
        for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
            exec_unit->papi_last_read[rv] = values[rv];
    }
    /* keep the contract with the previous registrant */
    if (compl_begin_prev != NULL) {
        (*compl_begin_prev)(exec_unit, exec_context, data);
    }
}

static void read_papi_core_complete_exec_count_end(dague_execution_unit_t * exec_unit,
                                                  dague_execution_context_t * exec_context,
                                                  void * data) {
    papi_core_exec_info_t info;
    info.kernel_type = -1;
    if (exec_context->dague_handle->profiling_array != NULL)
        info.kernel_type = exec_context->dague_handle->profiling_array[exec_context->function->function_id * 2] / 2;

    /* now count the PAPI events, if available */
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values);
    if (PAPI_OK != rv) {
        fprintf(stderr, "compl_end: couldn't read PAPI events in thread %d, ERROR: %s\n",
               exec_unit->th_id, PAPI_strerror(rv));
        for (rv = 0; rv < NUM_CORE_EVENTS; rv++)
            info.evt_values[rv] = 0;
    }
    else {
        for (rv = 0; rv < NUM_CORE_EVENTS; rv++)
            info.evt_values[rv] = values[rv] - exec_unit->papi_last_read[rv];
        for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
            exec_unit->papi_last_read[rv] = values[rv];
    }

    rv = dague_profiling_trace(exec_unit->eu_profile,
                               pins_prof_papi_core_compl_end,
                               31,
                               0,
                               (void *)&info);

    /* keep the contract with the previous registrant */
    if (compl_end_prev != NULL) {
        (*compl_end_prev)(exec_unit, exec_context, data);
    }
}
