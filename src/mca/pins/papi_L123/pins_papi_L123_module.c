#include <errno.h>
#include "dague_config.h"
#include "dague/mca/pins/pins.h"
#include "dague/mca/pins/pins_papi_utils.h"
#include "pins_papi_L123.h"
#include <papi.h>
#include <stdio.h>
#include "profiling.h"
#include "execution_unit.h"

static char* core_native_events   [NUM_CORE_EVENTS]   = PAPI_CORE_NATIVE_EVENT_ARRAY;
static char* socket_native_events [NUM_SOCKET_EVENTS] = PAPI_SOCKET_NATIVE_EVENT_ARRAY;

static int enable_exec = ENABLE_EXEC;
static int enable_select = ENABLE_SELECT;
static int enable_compl = ENABLE_COMPL;

static void pins_init_papi_L123(dague_context_t * master_context);
static void pins_fini_papi_L123(dague_context_t * master_context);
static void pins_thread_init_papi_L123(dague_execution_unit_t * exec_unit);
static void pins_thread_fini_papi_L123(dague_execution_unit_t * exec_unit);

static void read_papi_core_exec_count_begin(dague_execution_unit_t * exec_unit,
                                            dague_execution_context_t * exec_context,
                                            void * data);
static void read_papi_core_exec_count_end(dague_execution_unit_t * exec_unit,
                                          dague_execution_context_t * exec_context,
                                          void * data);
static void read_papi_core_select_count_begin(dague_execution_unit_t * exec_unit,
                                              dague_execution_context_t * exec_context,
                                              void * data);
static void read_papi_core_select_count_end(dague_execution_unit_t * exec_unit,
                                            dague_execution_context_t * exec_context,
                                            void * data);
static void read_papi_core_complete_exec_count_begin(dague_execution_unit_t * exec_unit,
                                                     dague_execution_context_t * exec_context,
                                                     void * data);
static void read_papi_core_complete_exec_count_end(dague_execution_unit_t * exec_unit,
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

static parsec_pins_callback * exec_begin_prev = NULL; // courtesy calls to previously-registered cbs
static parsec_pins_callback * exec_end_prev = NULL;
static parsec_pins_callback * select_begin_prev = NULL; // courtesy calls to previously-registered cbs
static parsec_pins_callback * select_end_prev = NULL;
static parsec_pins_callback * compl_begin_prev = NULL; // courtesy calls to previously-registered cbs
static parsec_pins_callback * compl_end_prev = NULL;
/*
 static parsec_pins_callback * thread_init_prev; // courtesy calls to previously-registered cbs
 static parsec_pins_callback * thread_fini_prev;
 */

static int pins_prof_papi_core_exec_begin,
    pins_prof_papi_core_exec_end,
    pins_prof_papi_core_select_begin,
    pins_prof_papi_core_select_end,
    pins_prof_papi_core_compl_begin,
    pins_prof_papi_core_compl_end,
    pins_prof_papi_socket_begin,
    pins_prof_papi_socket_end;

static void pins_init_papi_L123(dague_context_t * master_context) {
    pins_papi_init(master_context);
    /*
     thread_init_prev = PINS_REGISTER(THREAD_INIT, start_papi_L123);
     thread_fini_prev = PINS_REGISTER(THREAD_FINI, stop_papi_L123);
     */
    dague_profiling_add_dictionary_keyword(PAPI_CORE_PROF_EVT_NAME_SOCKET, "fill:#00AAFF",
                                           sizeof(papi_core_socket_info_t), NULL,
                                           &pins_prof_papi_socket_begin,
                                           &pins_prof_papi_socket_end);

    if (enable_exec) {
        exec_begin_prev = PINS_REGISTER(EXEC_BEGIN, read_papi_core_exec_count_begin);
        exec_end_prev   = PINS_REGISTER(EXEC_END, read_papi_core_exec_count_end);
        dague_profiling_add_dictionary_keyword(PAPI_CORE_PROF_EVT_NAME_EXEC, "fill:#00FF00",
                                               sizeof(papi_core_exec_info_t), NULL,
                                               &pins_prof_papi_core_exec_begin,
                                               &pins_prof_papi_core_exec_end);
    }
    if (enable_select) {
        select_begin_prev = PINS_REGISTER(SELECT_BEGIN, read_papi_core_select_count_begin);
        select_end_prev   = PINS_REGISTER(SELECT_END, read_papi_core_select_count_end);
        dague_profiling_add_dictionary_keyword(PAPI_CORE_PROF_EVT_NAME_SEL, "fill:#FFAA00",
                                               sizeof(papi_core_select_info_t), NULL,
                                               &pins_prof_papi_core_select_begin,
                                               &pins_prof_papi_core_select_end);
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
    PINS_REGISTER(EXEC_BEGIN,          exec_begin_prev);
    PINS_REGISTER(EXEC_END,            exec_end_prev);
    PINS_REGISTER(SELECT_BEGIN,        select_begin_prev);
    PINS_REGISTER(SELECT_END,          select_end_prev);
    PINS_REGISTER(COMPLETE_EXEC_BEGIN, compl_begin_prev);
    PINS_REGISTER(COMPLETE_EXEC_END,   compl_end_prev);
    /*
     PINS_REGISTER(THREAD_INIT, thread_init_prev);
     PINS_REGISTER(THREAD_FINI, thread_fini_prev);
     */
}

static void pins_thread_init_papi_L123(dague_execution_unit_t * exec_unit) {
    int rv = 0;
    int native;

    pins_papi_thread_init(exec_unit);

    exec_unit->papi_eventsets[EXEC_SET] = PAPI_NULL;
    if ((rv = PAPI_create_eventset(&exec_unit->papi_eventsets[EXEC_SET])) != PAPI_OK)
        printf("pins_papi_L123_module.c, pins_thread_init_papi_L123: "
               "thread %d couldn't create the PAPI event set "
               "to measure PAPI events; ERROR: %s\n",
               exec_unit->th_id, PAPI_strerror(rv));
    else {
        int i = 0;
        for (; i < NUM_CORE_EVENTS; i++) {
            if      (PAPI_OK != PAPI_event_name_to_code( core_native_events[i],
                                                         &native)) {
                /* error */
                printf("papi_L123 thread %d couldn't find event %s.\n",
                       exec_unit->th_id,
                       core_native_events[i]);
            }
            else if (PAPI_OK != PAPI_add_event(          exec_unit->papi_eventsets[EXEC_SET],
                                                         native)) {
                /* error */
                printf("papi_L123 thread %d couldn't add event %s.\n",
                       exec_unit->th_id,
                       core_native_events[i]);
            }
            else {
                /* printf("papi_L123 thread %d added event %s\n", */
                /*        exec_unit->th_id, */
                /*        core_native_events[i]); */
            }
        }
        if (exec_unit->th_id % CORES_PER_SOCKET == WHICH_CORE_IN_SOCKET) {
            for (i = 0; i < NUM_SOCKET_EVENTS; i++) {
                if ( PAPI_OK != PAPI_event_name_to_code( socket_native_events[i],
                                                         &native)) {
                    /* error */
                    printf("papi_L123 thread %d couldn't find event %s.\n",
                           exec_unit->th_id,
                           socket_native_events[i]);
                }
                else if (PAPI_OK !=
                         (rv = PAPI_add_event(exec_unit->papi_eventsets[EXEC_SET],
                                                         native))) {
                    /* error */
                    printf("papi_L123 thread %d couldn't add event %s, ERROR: %s\n",
                           exec_unit->th_id,
                           socket_native_events[i],
                           PAPI_strerror(rv));
                }
                else {
                    /* printf("papi_L123 thread %d added event %s\n", */
                    /*        exec_unit->th_id, */
                    /*        socket_native_events[i]); */
                }
            }
        }

        // start the event set (why wait?)
        if ((rv = PAPI_start(exec_unit->papi_eventsets[EXEC_SET])) != PAPI_OK)
            printf("papi_L123 couldn't start PAPI event set for thread %d; ERROR: %s\n",
                   exec_unit->th_id, PAPI_strerror(rv));
        else {
            rv = dague_profiling_trace(exec_unit->eu_profile,
                                       pins_prof_papi_socket_begin,
                                       48, 0, NULL);
        }
    }
}

static void pins_thread_fini_papi_L123(dague_execution_unit_t * exec_unit) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    if ((rv = PAPI_stop(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
        printf("papi_L123_module.c, pins_thread_fini_papi_L123: thread %d couldn't stop "
               "PAPI event set %d errno %d ERROR: %s\n",
               exec_unit->th_id, exec_unit->papi_eventsets[EXEC_SET],
               errno,
               PAPI_strerror(rv));
    }
    else {
        papi_core_socket_info_t info;
        for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
            info.evt_values[rv] = values[rv];
        rv = dague_profiling_trace(exec_unit->eu_profile,
                                   pins_prof_papi_socket_end,
                                   48, 0, (void *)&info);
    }
}

static void read_papi_core_exec_count_begin(dague_execution_unit_t * exec_unit,
                                           dague_execution_context_t * exec_context,
                                           void * data) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
        printf("exec_begin: couldn't read PAPI events in thread %d\n", exec_unit->th_id);
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
}

static void read_papi_core_exec_count_end(dague_execution_unit_t * exec_unit,
                                         dague_execution_context_t * exec_context,
                                         void * data) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
        printf("exec_end: couldn't read PAPI events in thread %d\n", exec_unit->th_id);
    }
    else {
        papi_core_exec_info_t info;
        info.kernel_type = -1;
        if (exec_context->dague_handle->profiling_array != NULL)
            info.kernel_type = exec_context->dague_handle->profiling_array[exec_context->function->function_id * 2] / 2;
        for (rv = 0; rv < NUM_CORE_EVENTS; rv++)
            info.evt_values[rv] = values[rv] - exec_unit->papi_last_read[rv];

        rv = dague_profiling_trace(exec_unit->eu_profile, pins_prof_papi_core_exec_end,
                                   (*exec_context->function->key
                                    )(exec_context->dague_handle, exec_context->locals),
                                   exec_context->dague_handle->handle_id,
                                   (void *)&info);
        for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
            exec_unit->papi_last_read[rv] = values[rv];
    }
    // keep the contract with the previous registrant
    if (exec_end_prev != NULL) {
        (*exec_end_prev)(exec_unit, exec_context, data);
    }
}

static void read_papi_core_select_count_begin(dague_execution_unit_t * exec_unit,
                                             dague_execution_context_t * exec_context,
                                             void * data) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
        printf("select_begin: couldn't read PAPI events in thread %d, ERROR: %s\n",
               exec_unit->th_id, PAPI_strerror(rv));
    }
    else {
        rv = dague_profiling_trace(exec_unit->eu_profile,
                                   pins_prof_papi_core_select_begin,
                                   32,
                                   0,
                                   (void *)NULL);
        for (rv = 0; rv < NUM_CORE_EVENTS + NUM_SOCKET_EVENTS; rv++)
            exec_unit->papi_last_read[rv] = values[rv];
    }
    // keep the contract with the previous registrant
    if (select_begin_prev != NULL) {
        (*select_begin_prev)(exec_unit, exec_context, data);
    }
}

static void read_papi_core_select_count_end(dague_execution_unit_t * exec_unit,
                                           dague_execution_context_t * exec_context,
                                           void * data) {
    unsigned long long victim_core_num = 0;
    unsigned int num_threads = (exec_unit->virtual_process->dague_context->nb_vp
                                * exec_unit->virtual_process->nb_cores);
    papi_core_select_info_t info;

    info.kernel_type = -1;
    if (exec_context) {
        victim_core_num = exec_context->victim_core;
        if (exec_context->dague_handle->profiling_array != NULL)
            info.kernel_type = exec_context->dague_handle->profiling_array[exec_context->function->function_id * 2] / 2;
        info.starvation = (unsigned long long)data;
    }
    info.victim_vp_id = -1; // currently unavailable from scheduler queue object
    if (victim_core_num >= num_threads)
        info.victim_vp_id = SYSTEM_QUEUE_VP;
    info.victim_th_id = (int)victim_core_num; // but this number includes the vp id multiplier
    info.exec_context = (unsigned long long int)exec_context; // if NULL, this was starvation

    // now count the PAPI events, if available
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
        printf("select_end: couldn't read PAPI events in thread %d, ERROR: %s\n",
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

    // test modification - do NOT trace starvation/steal misses
    rv = dague_profiling_trace(exec_unit->eu_profile,
                               pins_prof_papi_core_select_end,
                               32,
                               0,
                               (void *)&info);

    // keep the contract with the previous registrant
    if (select_end_prev != NULL) {
        (*select_end_prev)(exec_unit, exec_context, data);
    }
}

static void read_papi_core_complete_exec_count_begin(dague_execution_unit_t * exec_unit,
                                                    dague_execution_context_t * exec_context,
                                                    void * data) {
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
        printf("compl_begin: couldn't read PAPI events in thread %d, ERROR: %s\n",
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
    // keep the contract with the previous registrant
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

    // now count the PAPI events, if available
    int rv = PAPI_OK;
    long long int values[NUM_CORE_EVENTS + NUM_SOCKET_EVENTS];
    if ((rv = PAPI_read(exec_unit->papi_eventsets[EXEC_SET], values)) != PAPI_OK) {
        printf("compl_end: couldn't read PAPI events in thread %d, ERROR: %s\n",
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

    // keep the contract with the previous registrant
    if (compl_end_prev != NULL) {
        (*compl_end_prev)(exec_unit, exec_context, data);
    }
}
