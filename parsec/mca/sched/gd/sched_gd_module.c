/**
 * Copyright (c) 2013-2018 The University of Tennessee and The University
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
#include "parsec/utils/debug.h"
#include "parsec/mca/sched/sched.h"
#include "parsec/mca/sched/gd/sched_gd.h"
#include "parsec/class/dequeue.h"
#include "parsec/mca/pins/pins.h"

/**
 * Module functions
 */
static int sched_gd_install(parsec_context_t* master);
static int sched_gd_schedule(parsec_execution_stream_t* es,
                             parsec_task_t* new_context,
                             int32_t distance);
static parsec_task_t*
sched_gd_select(parsec_execution_stream_t *es,
                int32_t* distance);
static int flow_gd_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier);
static void sched_gd_remove(parsec_context_t* master);
static void sched_gd_register_sde( parsec_execution_stream_t *es );

const parsec_sched_module_t parsec_sched_gd_module = {
    &parsec_sched_gd_component,
    {
        sched_gd_install,
        flow_gd_init,
        sched_gd_schedule,
        sched_gd_select,
        NULL,
        sched_gd_register_sde,
        sched_gd_remove
    }
};

typedef struct {
    parsec_dequeue_t *dequeue;
    int               local_counter;
} shared_dequeue_with_local_counter_t;
#define LOCAL_SCHED_OBJECT(eu_context) ((shared_dequeue_with_local_counter_t*)(eu_context)->scheduler_object)

static long long int parsec_shared_dequeue_length( parsec_vp_t *vp )
{
    int thid;
    long long int sum = 0;

    for(thid = 0; thid < vp->nb_cores; thid++) {
        sum += LOCAL_SCHED_OBJECT(vp->execution_streams[thid])->local_counter;
    }
    return sum;
}

static int sched_gd_install( parsec_context_t *master )
{
    (void)master;
    return 0;
}

static void sched_gd_register_sde( parsec_execution_stream_t *es )
{
    char event_name[256];
    /* We register the counters only if the scheduler is installed, and only once per es */
    if( NULL != es && 0 == es->th_id ) {
        snprintf(event_name, 256, "PARSEC::SCHEDULER::PENDING_TASKS::QUEUE=%d::SCHED=GD", es->virtual_process->vp_id);
        papi_sde_register_fp_counter(parsec_papi_sde_handle, event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                     PAPI_SDE_int, (papi_sde_fptr_t)parsec_shared_dequeue_length, es->virtual_process);
        papi_sde_add_counter_to_group(parsec_papi_sde_handle, event_name,
                                      "PARSEC::SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
        papi_sde_add_counter_to_group(parsec_papi_sde_handle, event_name,
                                      "PARSEC::SCHEDULER::PENDING_TASKS::SCHED=GD", PAPI_SDE_SUM);
    }
    /* We describe the counters once if the scheduler is installed, or if we are called without
     * an execution stream (typically during papi_native_avail library load) */
    if( NULL == es || 0 == es->th_id ) {
        papi_sde_describe_counter(parsec_papi_sde_handle, "PARSEC::SCHEDULER::PENDING_TASKS::SCHED=GD",
                                  "the number of pending tasks for the GD scheduler");
        papi_sde_describe_counter(parsec_papi_sde_handle, "PARSEC::SCHEDULER::PENDING_TASKS::QUEUE=<VPID>::SCHED=GD",
                                  "the number of pending tasks for the GD scheduler on virtual process <VPID>");
    }
}

static int flow_gd_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier)
{
    parsec_vp_t *vp = es->virtual_process;

    /*
     * This function is called for each execution stream. However, as there is
     * a single global dequeue per context, it will be associated with the
     * first execution stream of the first virtual process. Every other
     * execution stream will make reference to the same dequeue (once we
     * succesfully synchronized all execution streams).
     */
    shared_dequeue_with_local_counter_t *sd = (shared_dequeue_with_local_counter_t*)calloc(sizeof(shared_dequeue_with_local_counter_t), 1);
    es->scheduler_object = sd;
    if (es == vp->execution_streams[0])
        sd->dequeue = OBJ_NEW(parsec_dequeue_t);

    parsec_barrier_wait(barrier);

    if (es != vp->execution_streams[0]) {
        sd->dequeue = LOCAL_SCHED_OBJECT(vp->execution_streams[0])->dequeue;
        OBJ_RETAIN(sd->dequeue);
    }
    
    sched_gd_register_sde( es );

    return 0;
}

static parsec_task_t*
sched_gd_select(parsec_execution_stream_t *es,
                int32_t* distance)
{
    shared_dequeue_with_local_counter_t *sd = LOCAL_SCHED_OBJECT(es);
    parsec_task_t * context =
        (parsec_task_t*)parsec_dequeue_try_pop_front( sd->dequeue );
    if(NULL != context)
        sd->local_counter--;
    *distance = 0;
    return context;
}

static int sched_gd_schedule(parsec_execution_stream_t* es,
                             parsec_task_t* new_context,
                             int32_t distance)
{
    int len = 0;
    shared_dequeue_with_local_counter_t *sd = LOCAL_SCHED_OBJECT(es);

    _LIST_ITEM_ITERATOR(new_context, &new_context->super, item, {len++; });

    if( (new_context->task_class->flags & PARSEC_HIGH_PRIORITY_TASK) &&
        (0 == distance) ) {
        parsec_dequeue_chain_front( sd->dequeue, (parsec_list_item_t*)new_context);
    } else {
        parsec_dequeue_chain_back( sd->dequeue, (parsec_list_item_t*)new_context);
    }

    sd->local_counter += len;
    return 0;
}

static void sched_gd_remove( parsec_context_t *master )
{
    shared_dequeue_with_local_counter_t *sd;
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    int p, t;
    char event_name[256];

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            sd = LOCAL_SCHED_OBJECT(es);
            OBJ_RELEASE( sd->dequeue );
            free(sd);
            es->scheduler_object = NULL;
        }
        snprintf(event_name, 256, "PARSEC::SCHEDULER::PENDING_TASKS::QUEUE=%d::SCHED=GD", p);
        papi_sde_unregister_counter(parsec_papi_sde_handle, event_name);
    }
    papi_sde_unregister_counter(parsec_papi_sde_handle, "PARSEC::SCHEDULER::PENDING_TASKS::SCHED=GD");
}
