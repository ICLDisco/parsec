/**
 * Copyright (c) 2018-2022 The University of Tennessee and The University
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
#include "parsec/mca/termdet/fourcounter/termdet_fourcounter.h"
#include "parsec/remote_dep.h"

/**
 * Module functions
 */

static void parsec_termdet_fourcounter_monitor_taskpool(parsec_taskpool_t *tp,
                                                        parsec_termdet_termination_detected_function_t cb);
static void parsec_termdet_fourcounter_unmonitor_taskpool(parsec_taskpool_t *tp);
static parsec_termdet_taskpool_state_t parsec_termdet_fourcounter_taskpool_state(parsec_taskpool_t *tp);
static int parsec_termdet_fourcounter_taskpool_ready(parsec_taskpool_t *tp);
static int parsec_termdet_fourcounter_taskpool_addto_nb_tasks(parsec_taskpool_t *tp, int v);
static int parsec_termdet_fourcounter_taskpool_addto_runtime_actions(parsec_taskpool_t *tp, int v);
static int parsec_termdet_fourcounter_taskpool_set_nb_tasks(parsec_taskpool_t *tp, int v);
static int parsec_termdet_fourcounter_taskpool_set_runtime_actions(parsec_taskpool_t *tp, int v);

static int parsec_termdet_fourcounter_outgoing_message_pack(parsec_taskpool_t *tp,
                                                            int dst_rank,
                                                            char *packed_buffer,
                                                            int *position,
                                                            int buffer_size);
static int parsec_termdet_fourcounter_outgoing_message_start(parsec_taskpool_t *tp,
                                                            int dst_rank,
                                                            parsec_remote_deps_t *remote_deps);
static int parsec_termdet_fourcounter_incoming_message_start(parsec_taskpool_t *tp,
                                                             int src_rank,
                                                             char *packed_buffer,
                                                             int *position,
                                                             int buffer_size,
                                                             const parsec_remote_deps_t *msg);
static int parsec_termdet_fourcounter_incoming_message_end(parsec_taskpool_t *tp,
                                                           const parsec_remote_deps_t *msg);
static int parsec_termdet_fourcounter_write_stats(parsec_taskpool_t *tp, FILE *fp);

const parsec_termdet_module_t parsec_termdet_fourcounter_module = {
    &parsec_termdet_fourcounter_component,
    {
        parsec_termdet_fourcounter_monitor_taskpool,
        parsec_termdet_fourcounter_unmonitor_taskpool,
        parsec_termdet_fourcounter_taskpool_state,
        parsec_termdet_fourcounter_taskpool_ready,
        parsec_termdet_fourcounter_taskpool_addto_nb_tasks,
        parsec_termdet_fourcounter_taskpool_addto_runtime_actions,
        parsec_termdet_fourcounter_taskpool_set_nb_tasks,
        parsec_termdet_fourcounter_taskpool_set_runtime_actions,
        0,
        parsec_termdet_fourcounter_outgoing_message_start,
        parsec_termdet_fourcounter_outgoing_message_pack,
        parsec_termdet_fourcounter_incoming_message_start,
        parsec_termdet_fourcounter_incoming_message_end,
        parsec_termdet_fourcounter_write_stats
    }
};

typedef enum {
    PARSEC_TERMDET_FOURCOUNTER_NOT_READY,
    PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_CHILDREN,
    PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_PARENT,
    PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_CHILDREN,
    PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_PARENT,
    PARSEC_TERMDET_FOURCOUNTER_TERMINATED
} parsec_termdet_fourcounter_state_t;

typedef struct parsec_termdet_fourcounter_monitor_s {
    parsec_atomic_rwlock_t rw_lock;             /**< Operations that change the state take the write lock, operations that
                                                 *   read the state take the read lock */
    uint32_t messages_sent;                     /**< Since the beginning, on that taskpool, how many messages have been sent */
    uint32_t messages_received;                 /**< Since the beginning, on that taskpool, how many messages have been received */
    parsec_termdet_fourcounter_state_t state;   /**< Current status */
    uint32_t nb_child_left;                     /**< If waiting for children, not ready, or busy, how many children have not provided their contribution yet */
    uint32_t acc_sent;                          /**< Accumulator for messages sent (sum of children when they contribute, plus this process
                                                 *   when the condition to switch to WAITING_FOR_PARENT is met */
    uint32_t acc_received;                      /**< Accumulator for messages received (sum of children when they contribute, plus this process
                                                 *   when the condition to switch to WAITING_FOR_PARENT is met */
    uint32_t last_acc_sent_at_root;             /**< Only for the root of the tree: how many sent messages were counted last time a decision was broadcast */
    uint32_t last_acc_received_at_root;         /**< Only for the root of the tree: how many received messages were counted last time a decision was broadcast */

    uint32_t stats_nb_busy_idle;                /**< Statistics: number of transitions busy -> idle */
    uint32_t stats_nb_idle_busy;                /**< Statistics: number of transitions idle -> busy */
    uint32_t stats_nb_sent_msg;                 /**< Statistics: number of messages sent */
    uint32_t stats_nb_recv_msg;                 /**< Statistics: number of messages received */
    uint32_t stats_nb_sent_bytes;               /**< Statistics: number of bytes sent */
    uint32_t stats_nb_recv_bytes;               /**< Statistics: number of bytes received */
    struct timeval stats_time_start;
    struct timeval stats_time_last_idle;
    struct timeval stats_time_end;
} parsec_termdet_fourcounter_monitor_t;


static void parsec_termdet_fourcounter_msg_down(parsec_termdet_fourcounter_msg_down_t *msg, int src, parsec_taskpool_t *tp);
static void parsec_termdet_fourcounter_msg_up(parsec_termdet_fourcounter_msg_up_t *msg, int src, parsec_taskpool_t *tp);

parsec_list_t parsec_termdet_fourcounter_delayed_messages;

static int parsec_termdet_fourcounter_msg_dispatch_taskpool(parsec_taskpool_t *tp, parsec_comm_engine_t *ce,
                                                            long unsigned int tag,  void *msg,
                                                            long unsigned int size, int src,  void *module)
{
    parsec_termdet_fourcounter_msg_type_t t = *(parsec_termdet_fourcounter_msg_type_t*)msg;
    parsec_termdet_fourcounter_msg_down_t *down_msg = (parsec_termdet_fourcounter_msg_down_t*)msg;
    parsec_termdet_fourcounter_msg_up_t *up_msg = (parsec_termdet_fourcounter_msg_up_t*)msg;
    (void)size;
    (void)tag;
    (void)module;
    (void)ce;

    parsec_list_unlock(&parsec_termdet_fourcounter_delayed_messages);
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tReceived %d bytes from %d relative to taskpool %d",
                         size, src, tp->taskpool_id);

    switch( t ) {
    case PARSEC_TERMDET_FOURCOUNTER_MSG_TYPE_DOWN:
        assert( size == sizeof(parsec_termdet_fourcounter_msg_down_t) );
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tIt is a DOWN message with result %d",
                             down_msg->result);
        parsec_termdet_fourcounter_msg_down( down_msg, src, tp );
        return PARSEC_SUCCESS;

    case PARSEC_TERMDET_FOURCOUNTER_MSG_TYPE_UP:
        assert( size == sizeof(parsec_termdet_fourcounter_msg_up_t) );
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tIt is an UP message with nb_sent = %d / nb_received = %d",
                             up_msg->nb_sent, up_msg->nb_received);
        parsec_termdet_fourcounter_msg_up( up_msg, src, tp );
        return PARSEC_SUCCESS;
    }
    assert(0);
    return PARSEC_ERROR;
}

int parsec_termdet_fourcounter_msg_dispatch(parsec_comm_engine_t *ce, parsec_ce_tag_t tag,  void *msg,
                                             size_t size, int src,  void *module)
{
    parsec_termdet_fourcounter_delayed_msg_t *delayed_msg;
    parsec_termdet_fourcounter_msg_down_t *down_msg = (parsec_termdet_fourcounter_msg_down_t*)msg;
    parsec_taskpool_t *tp = parsec_taskpool_lookup(down_msg->tp_id);

    if( (NULL == tp) || (NULL == tp->tdm.monitor) ||
        (((parsec_termdet_fourcounter_monitor_t*)tp->tdm.monitor)->state == PARSEC_TERMDET_FOURCOUNTER_NOT_READY) ) {
        parsec_list_lock(&parsec_termdet_fourcounter_delayed_messages);
        /* We re-check: somebody may have already inserted the
         * taskpool when we didn't have the lock */
        tp = parsec_taskpool_lookup(down_msg->tp_id);
        if ((NULL == tp) || (NULL == tp->tdm.monitor) ||
            (((parsec_termdet_fourcounter_monitor_t *) tp->tdm.monitor)->state ==
             PARSEC_TERMDET_FOURCOUNTER_NOT_READY)) {
            delayed_msg = (parsec_termdet_fourcounter_delayed_msg_t *) calloc(1,
                    sizeof(parsec_termdet_fourcounter_delayed_msg_t));
            PARSEC_LIST_ITEM_SINGLETON(delayed_msg);
            assert(size <= PARSEC_TERMDET_FOURCOUNTER_MAX_MSG_SIZE);
            delayed_msg->ce = ce;
            delayed_msg->module = module;
            delayed_msg->tag = tag;
            delayed_msg->size = size;
            delayed_msg->src = src;
            memcpy(delayed_msg->msg, msg, size);
            parsec_list_nolock_push_back(&parsec_termdet_fourcounter_delayed_messages, &delayed_msg->list_item);
            parsec_list_unlock(&parsec_termdet_fourcounter_delayed_messages);
            return PARSEC_SUCCESS;
        }
        parsec_list_unlock(&parsec_termdet_fourcounter_delayed_messages);
    }

    return parsec_termdet_fourcounter_msg_dispatch_taskpool(tp, ce, tag,  msg, size, src,  module);
}


static int parsec_termdet_fourcounter_topology_nb_children(parsec_taskpool_t *tp)
{
    parsec_context_t *context;
    assert(tp->context != NULL);
    context = tp->context;

    if( 2*context->my_rank + 2 < context->nb_nodes )
        return 2;
    if( 2*context->my_rank + 1 < context->nb_nodes )
        return 1;
    return 0;
}

static int parsec_termdet_fourcounter_topology_is_root(parsec_taskpool_t *tp)
{
    parsec_context_t *context;
    assert(tp->context != NULL);
    context = tp->context;
    return context->my_rank == 0;
}

static int parsec_termdet_fourcounter_topology_child(parsec_taskpool_t *tp, int i)
{
    parsec_context_t *context;
    assert(tp->context != NULL);
    context = tp->context;

    assert(i == 0 || i == 1);
    assert( 2*context->my_rank + i + 1 < context->nb_nodes);
    return 2 * context->my_rank + i + 1;
}

static int parsec_termdet_fourcounter_topology_parent(parsec_taskpool_t *tp)
{
    parsec_context_t *context;
    assert(tp->context != NULL);
    context = tp->context;

    assert(context->my_rank > 0);
    return (context->my_rank-1) >> 1;
}

static void parsec_termdet_fourcounter_monitor_taskpool(parsec_taskpool_t *tp,
                                                        parsec_termdet_termination_detected_function_t cb)
{
    parsec_termdet_fourcounter_monitor_t *tpm;
    assert(&parsec_termdet_fourcounter_module.module == tp->tdm.module);
    tpm = (parsec_termdet_fourcounter_monitor_t*)malloc(sizeof(parsec_termdet_fourcounter_monitor_t));
    tp->tdm.callback = cb;
    tpm->messages_sent = 0;
    tpm->messages_received = 0;
    tpm->state = PARSEC_TERMDET_FOURCOUNTER_NOT_READY;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tProcess initializes state to NOT_READY");
    tpm->nb_child_left = -1;
    tpm->acc_sent = 0;
    tpm->acc_received = 0;
    tpm->last_acc_sent_at_root = -1;
    tpm->last_acc_received_at_root = -1;
    tp->tdm.monitor = tpm;

    tpm->stats_nb_busy_idle = 0;
    tpm->stats_nb_idle_busy = 0;
    tpm->stats_nb_sent_msg = 0;
    tpm->stats_nb_recv_msg = 0;
    tpm->stats_nb_sent_bytes = 0;
    tpm->stats_nb_recv_bytes = 0;

    tp->nb_tasks = 0;
    tp->nb_pending_actions = 0;

    parsec_atomic_rwlock_init(&tpm->rw_lock);
    gettimeofday(&tpm->stats_time_start, NULL);
}

static void parsec_termdet_fourcounter_unmonitor_taskpool(parsec_taskpool_t *tp)
{
    assert(tp->tdm.module == &parsec_termdet_fourcounter_module.module);
    parsec_termdet_fourcounter_monitor_t *tpm;
    tpm = tp->tdm.monitor;
    assert(NULL != tpm);
    assert(tpm->state == PARSEC_TERMDET_FOURCOUNTER_TERMINATED);
    free(tpm);
    tp->tdm.monitor  = NULL;
    tp->tdm.module   = NULL;
    tp->tdm.callback = NULL;
}

static parsec_termdet_taskpool_state_t parsec_termdet_fourcounter_taskpool_state(parsec_taskpool_t *tp)
{
    parsec_termdet_fourcounter_monitor_t *tpm;
    parsec_termdet_fourcounter_state_t state;
    if( tp->tdm.module == NULL )
        return PARSEC_TERM_TP_NOT_MONITORED;
    assert(tp->tdm.module == &parsec_termdet_fourcounter_module.module);
    tpm = tp->tdm.monitor;
    parsec_atomic_rwlock_rdlock(&tpm->rw_lock);
    state = tpm->state;
    parsec_atomic_rwlock_rdunlock(&tpm->rw_lock);
    switch(state) {
    case PARSEC_TERMDET_FOURCOUNTER_NOT_READY:
        return PARSEC_TERM_TP_NOT_READY;
    case PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_CHILDREN:
    case PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_PARENT:
        return PARSEC_TERM_TP_BUSY;
    case PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_CHILDREN:
    case PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_PARENT:
        return PARSEC_TERM_TP_IDLE;
    case PARSEC_TERMDET_FOURCOUNTER_TERMINATED:
        return PARSEC_TERM_TP_TERMINATED;
    }
    assert(0);
    return (parsec_termdet_taskpool_state_t)-1;
}

static int parsec_termdet_fourcounter_taskpool_ready(parsec_taskpool_t *tp)
{
    parsec_termdet_fourcounter_monitor_t *tpm;
    parsec_list_item_t *item, *next;
    parsec_termdet_fourcounter_delayed_msg_t *delayed_msg;
    parsec_termdet_fourcounter_msg_down_t *down_msg;

    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_fourcounter_module.module );
    tpm = (parsec_termdet_fourcounter_monitor_t*)tp->tdm.monitor;
    assert( tpm->state == PARSEC_TERMDET_FOURCOUNTER_NOT_READY );
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    tpm->nb_child_left = parsec_termdet_fourcounter_topology_nb_children(tp);
    tpm->state = PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_CHILDREN; /* This is true even if nb_children == 0:
                                                                        * we will go in WAITING_FOR_PARENT only after
                                                                        * we sent the UP message */
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tProcess changed state for BUSY (taskpool ready)");
    parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    parsec_mfence();

    parsec_list_lock(&parsec_termdet_fourcounter_delayed_messages);
    for(item = PARSEC_LIST_ITERATOR_FIRST(&parsec_termdet_fourcounter_delayed_messages);
        item != PARSEC_LIST_ITERATOR_END(&parsec_termdet_fourcounter_delayed_messages);
        item = next) {
        next = PARSEC_LIST_ITEM_NEXT(item);
        delayed_msg = (parsec_termdet_fourcounter_delayed_msg_t*)item;
        down_msg = (parsec_termdet_fourcounter_msg_down_t*)delayed_msg->msg;
        if(down_msg->tp_id == tp->taskpool_id) {
            parsec_list_nolock_remove(&parsec_termdet_fourcounter_delayed_messages, item);
            parsec_termdet_fourcounter_msg_dispatch_taskpool(tp, delayed_msg->ce, delayed_msg->tag,
                                                            delayed_msg->msg, delayed_msg->size,
                                                            delayed_msg->src, delayed_msg->module);
        }
    }
    parsec_list_unlock(&parsec_termdet_fourcounter_delayed_messages);

    return PARSEC_SUCCESS;
}

static void parsec_termdet_fourcounter_send_up_messages(parsec_termdet_fourcounter_monitor_t *tpm,
                                                        parsec_taskpool_t *tp)
{
    parsec_termdet_fourcounter_msg_up_t msg_up;
    parsec_termdet_fourcounter_msg_down_t msg_down;
    int i;

    tpm->acc_sent += tpm->messages_sent;
    tpm->acc_received += tpm->messages_received;
    tpm->nb_child_left = parsec_termdet_fourcounter_topology_nb_children(tp);

    if( parsec_termdet_fourcounter_topology_is_root(tp) ) {
        msg_down.msg_type = PARSEC_TERMDET_FOURCOUNTER_MSG_TYPE_DOWN;
        msg_down.tp_id = tp->taskpool_id;
        if( parsec_termdet_fourcounter_topology_nb_children(tp) == 0 ) {
            /* Special case of singleton: don't do two waves */
            assert(tpm->acc_sent == tpm->acc_received);
            msg_down.result = 1;
        } else {
            msg_down.result = (tpm->last_acc_sent_at_root == tpm->acc_sent) &&
                (tpm->last_acc_received_at_root == tpm->acc_received) &&
                (tpm->acc_sent == tpm->acc_received);
        }
        for(i = 0; i < parsec_termdet_fourcounter_topology_nb_children(tp); i++) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tSending DOWN message with result %d to rank %d. Justification: last_acc_sent_at_root = %d, acc_sent = %d, last_acc_received_at_root = %d, acc_received = %d",
                                 msg_down.result, parsec_termdet_fourcounter_topology_child(tp, i),
                                 tpm->last_acc_sent_at_root, tpm->acc_sent, tpm->last_acc_received_at_root, tpm->acc_received);
            tpm->stats_nb_sent_msg++;
            tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_fourcounter_msg_down_t) + sizeof(int);
            parsec_ce.send_am(&parsec_ce, PARSEC_TERMDET_FOURCOUNTER_MSG_TAG, parsec_termdet_fourcounter_topology_child(tp, i), &msg_down, sizeof(parsec_termdet_fourcounter_msg_down_t));
        }
        tpm->last_acc_sent_at_root = tpm->acc_sent;
        tpm->last_acc_received_at_root = tpm->acc_received;
        if( msg_down.result ) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tTermination detected on root decision");
            gettimeofday(&tpm->stats_time_end, NULL);
            tpm->state = PARSEC_TERMDET_FOURCOUNTER_TERMINATED;
            tp->tdm.callback(tp);
        } else {
            tpm->acc_sent = 0;
            tpm->acc_received = 0;
        }
    } else {
        tpm->state = PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_PARENT;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tProcess changed state for IDLE_WAITING_FOR_PARENT");
        msg_up.msg_type = PARSEC_TERMDET_FOURCOUNTER_MSG_TYPE_UP;
        msg_up.tp_id = tp->taskpool_id;
        msg_up.nb_sent = tpm->acc_sent;
        msg_up.nb_received = tpm->acc_received;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tSending UP message with nb_sent / nb_received of %d/%d to rank %d",
                             msg_up.nb_sent, msg_up.nb_received, parsec_termdet_fourcounter_topology_parent(tp));
        tpm->stats_nb_sent_msg++;
        tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_fourcounter_msg_up_t) + sizeof(int);
        parsec_ce.send_am(&parsec_ce, PARSEC_TERMDET_FOURCOUNTER_MSG_TAG, parsec_termdet_fourcounter_topology_parent(tp), &msg_up, sizeof(parsec_termdet_fourcounter_msg_up_t));
    }
}

static void parsec_termdet_fourcounter_check_state_message_received(parsec_termdet_fourcounter_monitor_t *tpm,
                                                                    parsec_taskpool_t *tp)
{
    if(tp->nb_tasks == 0 &&
       tp->nb_pending_actions == 0 &&
       tpm->state == PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_CHILDREN &&
       tpm->nb_child_left == 0) {
        parsec_termdet_fourcounter_send_up_messages(tpm, tp);
    }
}

static void parsec_termdet_fourcounter_check_state_workload_changed(parsec_termdet_fourcounter_monitor_t *tpm,
                                                                    parsec_taskpool_t *tp)
{
    if(tp->nb_tasks == 0 && tp->nb_pending_actions == 0) {
        /* We are now IDLE */
        gettimeofday(&tpm->stats_time_last_idle, NULL);
        if( tpm->state == PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_PARENT) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tProcess changed state for IDLE_WAITING_FOR_PARENT");
            tpm->state = PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_PARENT;
            tpm->stats_nb_busy_idle++;
        } else if( tpm->state == PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_CHILDREN ) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tProcess changed state for IDLE_WAITING_FOR_CHILDREN");
            tpm->state = PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_CHILDREN;
            tpm->stats_nb_busy_idle++;
            if( tpm->nb_child_left == 0 ) {
                parsec_termdet_fourcounter_send_up_messages(tpm, tp);
            }
        }
    } else {
        /* We are now BUSY */
        if( tpm->state == PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_CHILDREN ) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tProcess changed state for BUSY_WAITING_FOR_CHILDREN");
            tpm->state = PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_CHILDREN;
            tpm->stats_nb_idle_busy++;
        } else if (tpm->state == PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_PARENT ) {
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tProcess changed state for BUSY_WAITING_FOR_PARENT");
            tpm->state = PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_PARENT;
            tpm->stats_nb_idle_busy++;
        }
    }
}

static int parsec_termdet_fourcounter_taskpool_set_nb_tasks(parsec_taskpool_t *tp, int v)
{
    parsec_termdet_fourcounter_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_fourcounter_module.module );
    assert( v >= 0 );
    tpm = (parsec_termdet_fourcounter_monitor_t *)tp->tdm.monitor;
    assert( tpm->state != PARSEC_TERMDET_FOURCOUNTER_TERMINATED );
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    if( (int)tp->nb_tasks != v) {
        tp->nb_tasks = v;
        parsec_termdet_fourcounter_check_state_workload_changed(tpm, tp);
    }
    parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    return v;
}

static int parsec_termdet_fourcounter_taskpool_set_runtime_actions(parsec_taskpool_t *tp, int v)
{
    parsec_termdet_fourcounter_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_fourcounter_module.module );
    assert( v >= 0 );
    tpm = (parsec_termdet_fourcounter_monitor_t *)tp->tdm.monitor;
    assert( tpm->state != PARSEC_TERMDET_FOURCOUNTER_TERMINATED );
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    if( (int)tp->nb_pending_actions != v) {
        tp->nb_pending_actions = v;
        parsec_termdet_fourcounter_check_state_workload_changed(tpm, tp);
    }
    parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    return v;
}

static int parsec_termdet_fourcounter_taskpool_addto_nb_tasks(parsec_taskpool_t *tp, int v)
{
    int ret;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_fourcounter_module.module );
    if(v == 0)
        return tp->nb_tasks;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tNB_TASKS %d -> %d", tp->nb_tasks, tp->nb_tasks + v);
    int tmp = parsec_atomic_fetch_add_int32(&tp->nb_tasks, v);
    assert( ((parsec_termdet_fourcounter_monitor_t *)tp->tdm.monitor)->state != PARSEC_TERMDET_FOURCOUNTER_TERMINATED );
    ret = tmp + v;
    if (tmp == 0 || ret == 0) {
        parsec_termdet_fourcounter_monitor_t *tpm;
        tpm = (parsec_termdet_fourcounter_monitor_t *)tp->tdm.monitor;
        /* Slow path: our changes might cause a state change so take a lock and check */
        parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
        parsec_termdet_fourcounter_check_state_workload_changed(tpm, tp);
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    }
    return ret;
}

static int parsec_termdet_fourcounter_taskpool_addto_runtime_actions(parsec_taskpool_t *tp, int v)
{
    int ret;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_fourcounter_module.module );
    assert( ((parsec_termdet_fourcounter_monitor_t *)tp->tdm.monitor)->state != PARSEC_TERMDET_FOURCOUNTER_TERMINATED );
    if(v == 0)
        return tp->nb_pending_actions;
    PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tNB_PA %d -> %d", tp->nb_pending_actions, tp->nb_pending_actions + v);
    int tmp = parsec_atomic_fetch_add_int32(&tp->nb_pending_actions, v);
    ret = tmp + v;
    if (tmp == 0 || ret == 0) {
        parsec_termdet_fourcounter_monitor_t *tpm;
        tpm = (parsec_termdet_fourcounter_monitor_t *)tp->tdm.monitor;
        /* Slow path: our changes might cause a state change so take a lock and check */
        parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
        parsec_termdet_fourcounter_check_state_workload_changed(tpm, tp);
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    }
    return ret;
}

static int parsec_termdet_fourcounter_outgoing_message_start(parsec_taskpool_t *tp,
                                                             int dst_rank,
                                                             parsec_remote_deps_t *remote_deps)
{
    parsec_termdet_fourcounter_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_fourcounter_module.module );
    (void)dst_rank;
    (void)remote_deps;
    tpm = tp->tdm.monitor;
    assert( tpm->state != PARSEC_TERMDET_FOURCOUNTER_TERMINATED );
    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    tpm->messages_sent++;
    parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);

    return 1;
}

static int parsec_termdet_fourcounter_outgoing_message_pack(parsec_taskpool_t *tp,
                                                            int dst_rank,
                                                            char *packed_buffer,
                                                            int *position,
                                                            int buffer_size)
{
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_fourcounter_module.module );
    /* No piggybacking */
    (void)dst_rank;
    (void)packed_buffer;
    (void)position;
    (void)buffer_size;
    (void)tp;
    return PARSEC_SUCCESS;
}

static int parsec_termdet_fourcounter_incoming_message_start(parsec_taskpool_t *tp,
                                                             int src_rank,
                                                             char *packed_buffer,
                                                             int *position,
                                                             int buffer_size,
                                                             const parsec_remote_deps_t *msg)
{
    parsec_termdet_fourcounter_monitor_t *tpm;
    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_fourcounter_module.module );

    tpm = tp->tdm.monitor;
    assert( tpm->state > PARSEC_TERMDET_FOURCOUNTER_NOT_READY);
    assert( tpm->state != PARSEC_TERMDET_FOURCOUNTER_TERMINATED );

    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    /* If we were ready or more, we become busy */
    if( tpm->state == PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_CHILDREN ) {
        tpm->state = PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_CHILDREN;
        tpm->stats_nb_idle_busy++;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tProcess changed state for BUSY_WAITING_FOR_CHILDREN (message start)");
    } else if(tpm->state == PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_PARENT ) {
        tpm->state = PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_PARENT;
        tpm->stats_nb_idle_busy++;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tProcess changed state for BUSY_WAITING_FOR_PARENT (message start)");
    }
    parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);

    /* No piggybacking */
    (void)src_rank;
    (void)packed_buffer;
    (void)position;
    (void)buffer_size;
    (void)msg;

    return PARSEC_SUCCESS;
}

static int parsec_termdet_fourcounter_incoming_message_end(parsec_taskpool_t *tp,
                                                           const parsec_remote_deps_t *msg)
{
    parsec_termdet_fourcounter_monitor_t *tpm;
    (void)msg;

    assert( tp->tdm.module != NULL );
    assert( tp->tdm.module == &parsec_termdet_fourcounter_module.module );

    tpm = tp->tdm.monitor;
    assert( tpm->state != PARSEC_TERMDET_FOURCOUNTER_TERMINATED );

    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    assert( tpm->state > PARSEC_TERMDET_FOURCOUNTER_NOT_READY);
    tpm->messages_received++;
    parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);

    return PARSEC_SUCCESS;
}

static void parsec_termdet_fourcounter_msg_down(parsec_termdet_fourcounter_msg_down_t *msg, int src, parsec_taskpool_t *tp)
{
    int i;
    parsec_termdet_fourcounter_monitor_t *tpm;

    (void)src;

    assert(NULL != tp->tdm.module);
    assert(&parsec_termdet_fourcounter_module.module == tp->tdm.module);
    tpm = (parsec_termdet_fourcounter_monitor_t *)tp->tdm.monitor;
    assert( tpm->state != PARSEC_TERMDET_FOURCOUNTER_TERMINATED );

    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    tpm->stats_nb_recv_msg++;
    tpm->stats_nb_recv_bytes += sizeof(parsec_termdet_fourcounter_msg_down_t)+sizeof(int);
    assert(tpm->state == PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_PARENT ||
           tpm->state == PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_PARENT );
    assert((int)tpm->nb_child_left == parsec_termdet_fourcounter_topology_nb_children(tp));

    for(i = 0; i < parsec_termdet_fourcounter_topology_nb_children(tp); i++) {
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tSending DOWN message with result %d to rank %d",
                             msg->result, parsec_termdet_fourcounter_topology_child(tp, i));
        tpm->stats_nb_sent_msg++;
        tpm->stats_nb_sent_bytes += sizeof(parsec_termdet_fourcounter_msg_down_t) + sizeof(int);
        parsec_ce.send_am(&parsec_ce, PARSEC_TERMDET_FOURCOUNTER_MSG_TAG, parsec_termdet_fourcounter_topology_child(tp, i), msg, sizeof(parsec_termdet_fourcounter_msg_down_t));
    }

    if(msg->result) {
        assert(tpm->state == PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_PARENT);
        gettimeofday(&tpm->stats_time_end, NULL);
        tpm->state = PARSEC_TERMDET_FOURCOUNTER_TERMINATED;
        PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tTermination detected on DOWN(true) message");
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
        tp->tdm.callback(tp);
    } else {
        tpm->acc_sent = 0;
        tpm->acc_received = 0;
        if( tpm->state == PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_PARENT ) {
            tpm->state = PARSEC_TERMDET_FOURCOUNTER_IDLE_WAITING_FOR_CHILDREN;
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tChange state to IDLE_WAITING_FOR_CHILDREN on DOWN(false) message");
            parsec_termdet_fourcounter_check_state_message_received(tpm, tp); /* In case tpm->nb_child_left is 0 already */
        } else {
            assert(tpm->state == PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_PARENT);
            PARSEC_DEBUG_VERBOSE(10, parsec_debug_output, "TERMDET-4C:\tChange state to BUSY_WAITING_FOR_CHILDREN on DOWN(false) message");
            tpm->state = PARSEC_TERMDET_FOURCOUNTER_BUSY_WAITING_FOR_CHILDREN;
        }
        parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
    }
}

static void parsec_termdet_fourcounter_msg_up(parsec_termdet_fourcounter_msg_up_t *msg, int src, parsec_taskpool_t *tp)
{
    parsec_termdet_fourcounter_monitor_t *tpm;

    (void)src;

    assert(NULL != tp->tdm.module);
    assert(&parsec_termdet_fourcounter_module.module == tp->tdm.module);
    tpm = (parsec_termdet_fourcounter_monitor_t *)tp->tdm.monitor;
    assert( tpm->state != PARSEC_TERMDET_FOURCOUNTER_TERMINATED );

    parsec_atomic_rwlock_wrlock(&tpm->rw_lock);
    tpm->stats_nb_recv_msg++;
    tpm->stats_nb_recv_bytes += sizeof(parsec_termdet_fourcounter_msg_up_t)+sizeof(int);
    assert( tpm->nb_child_left > 0 );

    tpm->acc_received += msg->nb_received;
    tpm->acc_sent += msg->nb_sent;
    tpm->nb_child_left--;

    parsec_termdet_fourcounter_check_state_message_received(tpm, tp);
    parsec_atomic_rwlock_wrunlock(&tpm->rw_lock);
}

static int parsec_termdet_fourcounter_write_stats(parsec_taskpool_t *tp, FILE *fp)
{
    parsec_termdet_fourcounter_monitor_t *tpm;
    struct timeval t1, t2;
    assert(NULL != tp->tdm.module);
    assert(&parsec_termdet_fourcounter_module.module == tp->tdm.module);
    tpm = (parsec_termdet_fourcounter_monitor_t *)tp->tdm.monitor;

    timersub(&tpm->stats_time_end, &tpm->stats_time_last_idle, &t1);
    timersub(&tpm->stats_time_end, &tpm->stats_time_start, &t2);

    fprintf(fp, "NP: %d M: 4C Rank: %d Taskpool#: %d #Transitions_Busy_to_Idle: %u #Transitions_Idle_to_Busy: %u #Times_Credit_was_Borrowed: 0 #Times_Credit_was_Flushed: 0 #Times_a_message_was_Delayed: 0 #Times_credit_was_merged: 0 #SentCtlMsg: %u #RecvCtlMsg: %u SentCtlBytes: %u RecvCtlBytes: %u WallTime: %u.%06u Idle2End: %u.%06u\n",
            tp->context->nb_nodes,
            tp->context->my_rank,
            tp->taskpool_id,
            tpm->stats_nb_busy_idle,
            tpm->stats_nb_idle_busy,
            tpm->stats_nb_sent_msg,
            tpm->stats_nb_recv_msg,
            tpm->stats_nb_sent_bytes,
            tpm->stats_nb_recv_bytes,
            (unsigned int)t2.tv_sec, (unsigned int)t2.tv_usec,
            (unsigned int)t1.tv_sec, (unsigned int)t1.tv_usec);

    return PARSEC_SUCCESS;
}
