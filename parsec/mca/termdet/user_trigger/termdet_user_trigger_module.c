/**
 * Copyright (c) 2021-2022 The University of Tennessee and The University
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
#include "parsec/mca/termdet/user_trigger/termdet_user_trigger.h"
#include "parsec/remote_dep.h"

/**
 * Module functions
 */

static void parsec_termdet_user_trigger_monitor_taskpool(parsec_taskpool_t *tp,
                                                         parsec_termdet_termination_detected_function_t cb);
static void parsec_termdet_user_trigger_unmonitor_taskpool(parsec_taskpool_t *tp);
static parsec_termdet_taskpool_state_t parsec_termdet_user_trigger_taskpool_state(parsec_taskpool_t *tp);
static int parsec_termdet_user_trigger_taskpool_ready(parsec_taskpool_t *tp);
static int parsec_termdet_user_trigger_taskpool_set_nb_tasks(parsec_taskpool_t *tp, int v);
static int parsec_termdet_user_trigger_taskpool_set_runtime_actions(parsec_taskpool_t *tp, int v);
static int parsec_termdet_user_trigger_taskpool_addto_nb_tasks(parsec_taskpool_t *tp, int v);
static int parsec_termdet_user_trigger_taskpool_addto_runtime_actions(parsec_taskpool_t *tp, int v);
static int parsec_termdet_user_trigger_outgoing_message_pack(parsec_taskpool_t *tp,
                                                             int dst_rank,
                                                             char *packed_buffer,
                                                             int *position,
                                                             int buffer_size);
static int parsec_termdet_user_trigger_outgoing_message_start(parsec_taskpool_t *tp,
                                                              int dst_rank,
                                                              parsec_remote_deps_t *remote_deps);
static int parsec_termdet_user_trigger_incoming_message_start(parsec_taskpool_t *tp,
                                                              int src_rank,
                                                              char *packed_buffer,
                                                              int *position,
                                                              int buffer_size,
                                                              const parsec_remote_deps_t *msg);
static int parsec_termdet_user_trigger_incoming_message_end(parsec_taskpool_t *tp,
                                                            const parsec_remote_deps_t *msg);

const parsec_termdet_module_t parsec_termdet_user_trigger_module = {
    &parsec_termdet_user_trigger_component,
    {
        parsec_termdet_user_trigger_monitor_taskpool,
        parsec_termdet_user_trigger_unmonitor_taskpool,
        parsec_termdet_user_trigger_taskpool_state,
        parsec_termdet_user_trigger_taskpool_ready,
        parsec_termdet_user_trigger_taskpool_addto_nb_tasks,
        parsec_termdet_user_trigger_taskpool_addto_runtime_actions,
        parsec_termdet_user_trigger_taskpool_set_nb_tasks,
        parsec_termdet_user_trigger_taskpool_set_runtime_actions,
        0,
        parsec_termdet_user_trigger_outgoing_message_start,
        parsec_termdet_user_trigger_outgoing_message_pack,
        parsec_termdet_user_trigger_incoming_message_start,
        parsec_termdet_user_trigger_incoming_message_end,
        NULL
    }
};

/* The root of the broadcast tree is only known when we set nb_tasks
 * to zero for the first time. Monitors should have the value below
 * until they know the root */
#define PARSEC_TERMDET_USER_TRIGGER_UNKNOWN_RANK (-1)

typedef enum {
    PARSEC_TERMDET_USER_TRIGGER_NOT_READY,
    PARSEC_TERMDET_USER_TRIGGER_BUSY,
    PARSEC_TERMDET_USER_TRIGGER_TERMINATED
} parsec_termdet_user_trigger_state_t;

typedef struct parsec_termdet_user_trigger_monitor_s {
    int32_t  root;                              /**< Who is the root of this bcast tree (known only when set_nb_tasks
                                                 *   is called */
    parsec_termdet_user_trigger_state_t state;  /**< Current status */
} parsec_termdet_user_trigger_monitor_t;

parsec_list_t parsec_termdet_user_trigger_delayed_messages;

static int parsec_termdet_user_trigger_msg_dispatch_taskpool(parsec_taskpool_t *tp, parsec_comm_engine_t *ce,
                                                             long unsigned int tag,  void *msg,
                                                             long unsigned int size, int src,  void *module)
{
    parsec_termdet_user_trigger_msg_t *ut_msg = (parsec_termdet_user_trigger_msg_t *)msg;
    parsec_termdet_user_trigger_monitor_t *monitor;

    assert(NULL != tp->tdm.monitor);
    monitor = (parsec_termdet_user_trigger_monitor_t*)tp->tdm.monitor;
    assert(PARSEC_TERMDET_USER_TRIGGER_TERMINATED != monitor->state);
    assert(PARSEC_TERMDET_USER_TRIGGER_UNKNOWN_RANK == monitor->root);

    (void)size;
    (void)tag;
    (void)module;
    (void)ce;
    (void)src;

    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tReceived %d bytes from %d relative to taskpool %d",
                         size, src, tp->taskpool_id);

    monitor->root = ut_msg->root;
    tp->tdm.module->taskpool_set_nb_tasks(tp, 0);

    return PARSEC_SUCCESS;
}

int parsec_termdet_user_trigger_msg_dispatch(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,  void *msg,
                                             size_t size, int src,  void *module)
{
    parsec_termdet_user_trigger_delayed_msg_t *delayed_msg;
    parsec_termdet_user_trigger_msg_t *ut_msg = (parsec_termdet_user_trigger_msg_t*)msg;
    parsec_taskpool_t *tp = parsec_taskpool_lookup(ut_msg->tp_id);

    assert((NULL == tp) || (PARSEC_TERMDET_USER_TRIGGER_TERMINATED != ((parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor)->state));

    if( (NULL == tp) || (NULL == tp->tdm.monitor) ||
        (((parsec_termdet_user_trigger_monitor_t*)tp->tdm.monitor)->state == PARSEC_TERMDET_USER_TRIGGER_NOT_READY) ) {
        parsec_list_lock(&parsec_termdet_user_trigger_delayed_messages);
        /* We re-check: somebody may have already inserted the
         * taskpool when we didn't have the lock */
        tp = parsec_taskpool_lookup(ut_msg->tp_id);
        if( (NULL == tp) || (NULL == tp->tdm.monitor) ||
            (((parsec_termdet_user_trigger_monitor_t*)tp->tdm.monitor)->state == PARSEC_TERMDET_USER_TRIGGER_NOT_READY)) {
            delayed_msg = (parsec_termdet_user_trigger_delayed_msg_t *) malloc(
                    sizeof(parsec_termdet_user_trigger_delayed_msg_t));
            PARSEC_LIST_ITEM_SINGLETON(delayed_msg);
            assert(size <= PARSEC_TERMDET_USER_TRIGGER_MAX_MSG_SIZE);
            delayed_msg->ce = ce;
            delayed_msg->module = module;
            delayed_msg->tag = tag;
            delayed_msg->size = size;
            delayed_msg->src = src;
            memcpy(delayed_msg->msg, msg, size);
            parsec_list_nolock_push_back(&parsec_termdet_user_trigger_delayed_messages, &delayed_msg->list_item);
            parsec_list_unlock(&parsec_termdet_user_trigger_delayed_messages);
            return PARSEC_SUCCESS;
        }
        parsec_list_unlock(&parsec_termdet_user_trigger_delayed_messages);
    }

    return parsec_termdet_user_trigger_msg_dispatch_taskpool(tp, ce, tag,  msg, size, src,  module);
}

static void parsec_termdet_user_trigger_monitor_taskpool(parsec_taskpool_t *tp,
                                                         parsec_termdet_termination_detected_function_t cb)
{
    parsec_termdet_user_trigger_monitor_t *monitor;
    assert(&parsec_termdet_user_trigger_module.module == tp->tdm.module);
    //assert(NULL == tp->tdm.monitor);
    monitor = malloc(sizeof(parsec_termdet_user_trigger_monitor_t));
    monitor->root = PARSEC_TERMDET_USER_TRIGGER_UNKNOWN_RANK;
    monitor->state = PARSEC_TERMDET_USER_TRIGGER_NOT_READY;
    tp->tdm.callback = cb;
    tp->tdm.monitor  = monitor;
    tp->nb_tasks     = PARSEC_UNDETERMINED_NB_TASKS;
}

static void parsec_termdet_user_trigger_unmonitor_taskpool(parsec_taskpool_t *tp)
{
    parsec_termdet_user_trigger_monitor_t *monitor;
    assert(&parsec_termdet_user_trigger_module.module == tp->tdm.module);
    monitor = (parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor;
    assert(NULL != monitor);
    assert(monitor->state == PARSEC_TERMDET_USER_TRIGGER_TERMINATED);
    free(monitor);
    tp->tdm.monitor = NULL;
    tp->tdm.module   = NULL;
    tp->tdm.callback = NULL;
}


static parsec_termdet_taskpool_state_t parsec_termdet_user_trigger_taskpool_state(parsec_taskpool_t *tp)
{
    if( tp->tdm.module == NULL )
        return PARSEC_TERM_TP_NOT_MONITORED;
    assert(tp->tdm.module == &parsec_termdet_user_trigger_module.module);
    if( ((parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor)->state == PARSEC_TERMDET_USER_TRIGGER_BUSY )
        return PARSEC_TERM_TP_BUSY;
    if( ((parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor)->state == PARSEC_TERMDET_USER_TRIGGER_NOT_READY )
        return PARSEC_TERM_TP_NOT_READY;
    if( ((parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor)->state == PARSEC_TERMDET_USER_TRIGGER_TERMINATED )
        return PARSEC_TERM_TP_TERMINATED;
    assert(0);
    return -1;
}

static int parsec_termdet_user_trigger_taskpool_ready(parsec_taskpool_t *tp)
{
    parsec_list_item_t *item, *next;
    parsec_termdet_user_trigger_delayed_msg_t *delayed_msg;
    parsec_termdet_user_trigger_msg_t *msg;

    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_user_trigger_module.module );
    assert( ((parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor)->state == PARSEC_TERMDET_USER_TRIGGER_NOT_READY );
    parsec_atomic_fetch_inc_int32(&tp->nb_pending_actions); // We count 'the tasks' as a pending action
    ((parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor)->state = PARSEC_TERMDET_USER_TRIGGER_BUSY;

    parsec_list_lock(&parsec_termdet_user_trigger_delayed_messages);
    for(item = PARSEC_LIST_ITERATOR_FIRST(&parsec_termdet_user_trigger_delayed_messages);
        item != PARSEC_LIST_ITERATOR_END(&parsec_termdet_user_trigger_delayed_messages);
        item = next) {
        next = PARSEC_LIST_ITEM_NEXT(item);
        delayed_msg = (parsec_termdet_user_trigger_delayed_msg_t*)item;
        msg = (parsec_termdet_user_trigger_msg_t*)delayed_msg->msg;
        if(msg->tp_id == tp->taskpool_id) {
            parsec_list_nolock_remove(&parsec_termdet_user_trigger_delayed_messages, item);
            parsec_termdet_user_trigger_msg_dispatch_taskpool(tp, delayed_msg->ce, delayed_msg->tag,
                                                             delayed_msg->msg, delayed_msg->size,
                                                             delayed_msg->src, delayed_msg->module);
        }
    }
    parsec_list_unlock(&parsec_termdet_user_trigger_delayed_messages);

    return PARSEC_SUCCESS;
}

static int32_t parsec_termdet_user_trigger_taskpool_set_nb_tasks(parsec_taskpool_t *tp, int32_t v)
{
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-USER_TRIGGER:\tNB_TASKS -> %d", v);
    if(v != 0) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-USER_TRIGGER:\tNB_TASKS -> %d ignored", v);
        return tp->nb_tasks;
    }
    assert(((parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor)->state == PARSEC_TERMDET_USER_TRIGGER_BUSY);
    assert(tp->nb_tasks == PARSEC_UNDETERMINED_NB_TASKS);
    if(((parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor)->root==PARSEC_TERMDET_USER_TRIGGER_UNKNOWN_RANK) {
        /* I am the root of the broadcast tree */
        parsec_termdet_user_trigger_monitor_t *monitor = (parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor;
        monitor->root = tp->context->my_rank;
    }
    tp->nb_tasks = 0;
    parsec_termdet_user_trigger_taskpool_addto_runtime_actions(tp, -1);
    return tp->nb_tasks;
}

static void parsec_termdet_signal_termination(parsec_taskpool_t *tp)
{
    parsec_termdet_user_trigger_monitor_t *monitor = (parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor;
    parsec_termdet_user_trigger_msg_t msg;

    monitor->state = PARSEC_TERMDET_USER_TRIGGER_TERMINATED;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-USER_TRIGGER:\tBUSY -> TERMINATED. Broadcast detection");

    msg.tp_id = tp->taskpool_id;
    msg.root  = monitor->root;

    // Simple broadcast with binary tree
    // TODO: use the parsec_ce broadcast API when available
    // my rank in the shifted world where monitor->root is 0
    int my_rank = (tp->context->my_rank - monitor->root + tp->context->nb_nodes) % tp->context->nb_nodes;
    int nb_children =  2*my_rank + 2 < tp->context->nb_nodes ? 2 : (2*my_rank + 1 < tp->context->nb_nodes ? 1 : 0 );
    for(int i = 0; i < nb_children; i++) {
        int child = 2 * my_rank + i + 1;
        int real_child = (child + monitor->root) % tp->context->nb_nodes;
        parsec_ce.send_am(&parsec_ce, PARSEC_TERMDET_USER_TRIGGER_MSG_TAG, real_child, &msg,
                          sizeof(parsec_termdet_user_trigger_msg_t));
    }

    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-USER_TRIGGER:\tcall callback");
    tp->tdm.callback(tp);
}


static int32_t parsec_termdet_user_trigger_taskpool_set_runtime_actions(parsec_taskpool_t *tp, int32_t v)
{
    int32_t ov;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-USER_TRIGGER:\tNB_PA -> %d", v);
    do {
        ov = tp->nb_pending_actions;
    } while(!parsec_atomic_cas_int32(&tp->nb_pending_actions, ov, v));

    if( ((parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor)->state == PARSEC_TERMDET_USER_TRIGGER_BUSY && v == 0 ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-USER_TRIGGER:\tnbpa set to 0");
        parsec_termdet_signal_termination(tp);
    }
    return v;
}

static int32_t parsec_termdet_user_trigger_taskpool_addto_nb_tasks(parsec_taskpool_t *tp, int32_t v)
{
    (void)v;
    return tp->nb_tasks;
}

static int32_t parsec_termdet_user_trigger_taskpool_addto_runtime_actions(parsec_taskpool_t *tp, int32_t v)
{
    int32_t ov;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-USER_TRIGGER:\tNB_PA %d -> %d", tp->nb_pending_actions,
                         tp->nb_pending_actions + v);
    if(v == 0)
        return tp->nb_pending_actions;
    ov = parsec_atomic_fetch_add_int32(&tp->nb_pending_actions, v);
    if( ((parsec_termdet_user_trigger_monitor_t *)tp->tdm.monitor)->state == PARSEC_TERMDET_USER_TRIGGER_BUSY &&
        ov+v == 0 ) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-USER_TRIGGER:\tnbpa == 0");
        parsec_termdet_signal_termination(tp);
    }
    return ov+v;
}

static int parsec_termdet_user_trigger_outgoing_message_start(parsec_taskpool_t *tp,
                                                              int dst_rank,
                                                              parsec_remote_deps_t *remote_deps)
{
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_user_trigger_module.module );
    /* Nothing to do with the message */
    (void)dst_rank;
    (void)remote_deps;
    (void)tp;
    return 1; /* The message can go right away */
}
static int parsec_termdet_user_trigger_outgoing_message_pack(parsec_taskpool_t *tp,
                                                             int dst_rank,
                                                             char *packed_buffer,
                                                             int *position,
                                                             int buffer_size)
{
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_user_trigger_module.module );
    /* No piggybacking */
    (void)dst_rank;
    (void)packed_buffer;
    (void)position;
    (void)buffer_size;
    (void)tp;
    return PARSEC_SUCCESS;
}

static int parsec_termdet_user_trigger_incoming_message_start(parsec_taskpool_t *tp,
                                                              int src_rank,
                                                              char *packed_buffer,
                                                              int *position,
                                                              int buffer_size,
                                                              const parsec_remote_deps_t *msg)
{
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_user_trigger_module.module );
    /* No piggybacking */
    (void)src_rank;
    (void)packed_buffer;
    (void)position;
    (void)buffer_size;
    (void)msg;
    (void)tp;
    return PARSEC_SUCCESS;
}

static int parsec_termdet_user_trigger_incoming_message_end(parsec_taskpool_t *tp,
                                                            const parsec_remote_deps_t *msg)
{
    (void)tp;
    (void)msg;
    return PARSEC_SUCCESS;
}
