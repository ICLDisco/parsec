/**
 * Copyright (c) 2018-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "parsec/parsec_config.h"
#include "parsec/parsec_internal.h"
#include "parsec/include/parsec/execution_stream.h"
#include "parsec/utils/debug.h"
#include "parsec/mca/termdet/termdet.h"
#include "parsec/mca/termdet/local/termdet_local.h"

/**
 * Module functions
 */

static void parsec_termdet_local_monitor_taskpool(parsec_taskpool_t *tp,
                                                  parsec_termdet_termination_detected_function_t cb);
static void parsec_termdet_local_unmonitor_taskpool(parsec_taskpool_t *tp);
static parsec_termdet_taskpool_state_t parsec_termdet_local_taskpool_state(parsec_taskpool_t *tp);
static int parsec_termdet_local_taskpool_ready(parsec_taskpool_t *tp);
static int parsec_termdet_local_taskpool_set_nb_tasks(parsec_taskpool_t *tp, int v);
static int parsec_termdet_local_taskpool_set_runtime_actions(parsec_taskpool_t *tp, int v);
static int parsec_termdet_local_taskpool_addto_nb_tasks(parsec_taskpool_t *tp, int v);
static int parsec_termdet_local_taskpool_addto_runtime_actions(parsec_taskpool_t *tp, int v);
static int parsec_termdet_local_outgoing_message_pack(parsec_taskpool_t *tp,
                                                      int dst_rank,
                                                      char *packed_buffer,
                                                      int *position,
                                                      int buffer_size);
static int parsec_termdet_local_outgoing_message_start(parsec_taskpool_t *tp,
                                                       int dst_rank,
                                                       parsec_remote_deps_t *remote_deps);
static int parsec_termdet_local_incoming_message_start(parsec_taskpool_t *tp,
                                                       int src_rank,
                                                       char *packed_buffer,
                                                       int *position,
                                                       int buffer_size,
                                                       const parsec_remote_deps_t *msg);
static int parsec_termdet_local_incoming_message_end(parsec_taskpool_t *tp,
                                                     const parsec_remote_deps_t *msg);

const parsec_termdet_module_t parsec_termdet_local_module = {
    &parsec_termdet_local_component,
    {
        parsec_termdet_local_monitor_taskpool,
        parsec_termdet_local_unmonitor_taskpool,
        parsec_termdet_local_taskpool_state,
        parsec_termdet_local_taskpool_ready,
        parsec_termdet_local_taskpool_addto_nb_tasks,
        parsec_termdet_local_taskpool_addto_runtime_actions,
        parsec_termdet_local_taskpool_set_nb_tasks,
        parsec_termdet_local_taskpool_set_runtime_actions,
        0,
        parsec_termdet_local_outgoing_message_start, 
        parsec_termdet_local_outgoing_message_pack,
        parsec_termdet_local_incoming_message_start,
        parsec_termdet_local_incoming_message_end,
        NULL
    }
};

/* The local detector does not need to allocate memory:
 * we use the constants below to keep track of the state.
 * There is no need for a constant for idle, as the termdet
 * transitions directly from busy to terminated.
 */
#define PARSEC_TERMDET_LOCAL_TERMINATED NULL
#define PARSEC_TERMDET_LOCAL_NOT_READY   ((void*)(0x1))
#define PARSEC_TERMDET_LOCAL_BUSY        ((void*)(0x2))

static void parsec_termdet_local_monitor_taskpool(parsec_taskpool_t *tp,
                                                  parsec_termdet_termination_detected_function_t cb)
{
    assert(&parsec_termdet_local_module.module == tp->tdm.module);
    tp->tdm.callback = cb;
    tp->tdm.monitor = PARSEC_TERMDET_LOCAL_NOT_READY;
}

static void parsec_termdet_local_unmonitor_taskpool(parsec_taskpool_t *tp)
{
    assert(&parsec_termdet_local_module.module == tp->tdm.module);
    assert(tp->tdm.monitor == PARSEC_TERMDET_LOCAL_TERMINATED);
    tp->tdm.module   = NULL;
    tp->tdm.callback = NULL;
}

static parsec_termdet_taskpool_state_t parsec_termdet_local_taskpool_state(parsec_taskpool_t *tp)
{
    if( tp->tdm.module == NULL )
        return PARSEC_TERM_TP_NOT_MONITORED;
    assert(tp->tdm.module == &parsec_termdet_local_module.module);
    if( tp->tdm.monitor == PARSEC_TERMDET_LOCAL_TERMINATED )
        return PARSEC_TERM_TP_TERMINATED;
    if( tp->tdm.monitor == PARSEC_TERMDET_LOCAL_BUSY )
        return PARSEC_TERM_TP_BUSY;
    if( tp->tdm.monitor == PARSEC_TERMDET_LOCAL_NOT_READY )
        return PARSEC_TERM_TP_NOT_READY;
    assert(0);
    return -1;
}

static void parsec_termdet_local_termination_detected(parsec_taskpool_t *tp)
{
    assert( tp->tdm.module != NULL);
    assert( tp->tdm.module == &parsec_termdet_local_module.module );
    assert( tp->tdm.monitor == PARSEC_TERMDET_LOCAL_TERMINATED );

    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL\tTASKPOOL %p: termination detected", tp);

    if(NULL != tp->tdm.callback) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL\tTASKPOOL %p: calling callback", tp);
        tp->tdm.callback(tp);
    }
}

static int parsec_termdet_local_taskpool_ready(parsec_taskpool_t *tp)
{
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_local_module.module );
    assert( tp->tdm.monitor == PARSEC_TERMDET_LOCAL_NOT_READY );
    parsec_atomic_cas_ptr(&tp->tdm.monitor, PARSEC_TERMDET_LOCAL_NOT_READY, PARSEC_TERMDET_LOCAL_BUSY);
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL:\tTASKPOOL %p READY", tp);

    if( tp->nb_pending_actions == 0) {
        /* It's possible another thread sees nb_pending_actions == 0 and BUSY before me, so call the callback
         * only if I'm the one setting to terminated */
        if( parsec_atomic_cas_ptr(&tp->tdm.monitor, PARSEC_TERMDET_LOCAL_BUSY, PARSEC_TERMDET_LOCAL_TERMINATED) ) {
            parsec_termdet_local_termination_detected(tp);
        }
    }
    return PARSEC_SUCCESS;
}

static int32_t parsec_termdet_local_taskpool_set_nb_tasks(parsec_taskpool_t *tp, int32_t v)
{
    int32_t ov, nbpa = 1; // By default we are not the one to discover nbpa gets to 0
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL:\tTASKPOOL %p NB_TASKS -> %d", tp, v);
    assert(v >= 0 || v == PARSEC_RUNTIME_RESERVED_NB_TASKS);
    if(tp->nb_tasks != v) {
        do {
            ov = tp->nb_tasks;
        } while(! parsec_atomic_cas_int32(&tp->nb_tasks, ov, v));
        if( ov == 0 && v > 0 ) {
            nbpa = parsec_atomic_fetch_inc_int32(&tp->nb_pending_actions) + 1;
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL:\tTASKPOOL %p  NB_PA %d -> %d", tp, nbpa-1, nbpa);
        } else if( ov > 0 && v == 0 ) {
            nbpa = parsec_atomic_fetch_dec_int32(&tp->nb_pending_actions) - 1;
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL:\tTASKPOOL %p  NB_PA %d -> %d", tp, nbpa+1, nbpa);
        }
        if( tp->tdm.monitor == PARSEC_TERMDET_LOCAL_BUSY && nbpa == 0 ) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL:\tTASKPOOL %p nbpa == 0", tp);
            if( parsec_atomic_cas_ptr(&tp->tdm.monitor, PARSEC_TERMDET_LOCAL_BUSY, PARSEC_TERMDET_LOCAL_TERMINATED) ) {
                parsec_termdet_local_termination_detected(tp);
            }
        }
    }
    return tp->nb_tasks;
}

static int32_t parsec_termdet_local_taskpool_set_runtime_actions(parsec_taskpool_t *tp, int32_t v)
{
    int32_t ov;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL:\tTASKPOOL %p NB_PA -> %d", tp, v);
    assert(v >= 0);
    do {
        ov = tp->nb_pending_actions;
    } while(!parsec_atomic_cas_int32(&tp->nb_pending_actions, ov, v));
    if( tp->tdm.monitor == PARSEC_TERMDET_LOCAL_BUSY && v == 0 ) {
        if( parsec_atomic_cas_ptr(&tp->tdm.monitor, PARSEC_TERMDET_LOCAL_BUSY, PARSEC_TERMDET_LOCAL_TERMINATED) ) {
            parsec_termdet_local_termination_detected(tp);
        }
    }
    return v;
}

static int32_t parsec_termdet_local_taskpool_addto_nb_tasks(parsec_taskpool_t *tp, int32_t v)
{
    int32_t ov, nbpa = 1; // By default we are not the one to discover that nbpa is 0
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL:\tTASKPOOL %p NB_TASKS %d -> %d", tp, tp->nb_tasks, tp->nb_tasks + v);
    if(v == 0)
        return tp->nb_tasks;
    ov = parsec_atomic_fetch_add_int32(&tp->nb_tasks, v);
    assert(ov + v >= 0);
    if(ov == 0 && v > 0) {
        nbpa = parsec_atomic_fetch_inc_int32(&tp->nb_pending_actions) + 1;
        assert(nbpa >= 0);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL:\tTASKPOOL %p  NB_PA %d -> %d", tp, nbpa-1, nbpa);
    } else if(ov + v == 0 && ov > 0) {
        nbpa = parsec_atomic_fetch_dec_int32(&tp->nb_pending_actions) - 1;
        assert(nbpa >= 0);
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL:\tTASKPOOL %p  NB_PA %d -> %d", tp, nbpa+1, nbpa);
    }
    if( tp->tdm.monitor == PARSEC_TERMDET_LOCAL_BUSY && nbpa == 0 ) {
        if( parsec_atomic_cas_ptr(&tp->tdm.monitor, PARSEC_TERMDET_LOCAL_BUSY, PARSEC_TERMDET_LOCAL_TERMINATED) ) {
            parsec_termdet_local_termination_detected(tp);
        }
    }
    return ov+v;
}

static int32_t parsec_termdet_local_taskpool_addto_runtime_actions(parsec_taskpool_t *tp, int32_t v)
{
    int32_t ov;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-LOCAL:\tTASKPOOL %p NB_PA %d -> %d", tp, tp->nb_pending_actions,
                         tp->nb_pending_actions + v);
    if(v == 0)
        return tp->nb_pending_actions;
    ov = parsec_atomic_fetch_add_int32(&tp->nb_pending_actions, v);
    assert(ov+v >= 0);
    if( tp->tdm.monitor == PARSEC_TERMDET_LOCAL_BUSY && ov+v == 0 ) {
        if( parsec_atomic_cas_ptr(&tp->tdm.monitor, PARSEC_TERMDET_LOCAL_BUSY, PARSEC_TERMDET_LOCAL_TERMINATED) ) {
            parsec_termdet_local_termination_detected(tp);
        }
    }
    return ov+v;
}

static int parsec_termdet_local_outgoing_message_start(parsec_taskpool_t *tp,
                                                      int dst_rank,
                                                      parsec_remote_deps_t *remote_deps)
{
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_local_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_LOCAL_TERMINATED );
    /* Nothing to do with the message */
    (void)dst_rank;
    (void)remote_deps;
    (void)tp;
    return 1; /* The message can go right away */
}

static int parsec_termdet_local_outgoing_message_pack(parsec_taskpool_t *tp,
                                                      int dst_rank,
                                                      char *packed_buffer,
                                                      int *position,
                                                      int buffer_size)
{
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_local_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_LOCAL_TERMINATED );
    /* No piggybacking */
    (void)dst_rank;
    (void)packed_buffer;
    (void)position;
    (void)buffer_size;
    (void)tp;
    return PARSEC_SUCCESS;
}

static int parsec_termdet_local_incoming_message_start(parsec_taskpool_t *tp,
                                                       int src_rank,
                                                       char *packed_buffer,
                                                       int *position,
                                                       int buffer_size,
                                                       const parsec_remote_deps_t *msg)
{
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_local_module.module );
    assert( tp->tdm.monitor != PARSEC_TERMDET_LOCAL_TERMINATED );
    /* No piggybacking */
    (void)src_rank;
    (void)packed_buffer;
    (void)position;
    (void)buffer_size;
    (void)msg;
    (void)tp;
    return PARSEC_SUCCESS;
}

static int parsec_termdet_local_incoming_message_end(parsec_taskpool_t *tp,
                                                     const parsec_remote_deps_t *msg)
{
    (void)tp;
    (void)msg;
    return PARSEC_SUCCESS;
}
