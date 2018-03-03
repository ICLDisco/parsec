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

const parsec_sched_module_t parsec_sched_gd_module = {
    &parsec_sched_gd_component,
    {
        sched_gd_install,
        flow_gd_init,
        sched_gd_schedule,
        sched_gd_select,
        NULL,
        sched_gd_remove
    }
};

static int sched_gd_install( parsec_context_t *master )
{
    (void)master;
    return 0;
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
    if (es == vp->execution_streams[0])
        vp->execution_streams[0]->scheduler_object = OBJ_NEW(parsec_dequeue_t);

    parsec_barrier_wait(barrier);

    if (es != vp->execution_streams[0]) {
        es->scheduler_object = (void*)vp->execution_streams[0]->scheduler_object;
        OBJ_RETAIN(es->scheduler_object);
    }
    return 0;
}

static parsec_task_t*
sched_gd_select(parsec_execution_stream_t *es,
                int32_t* distance)
{
    parsec_task_t * context =
        (parsec_task_t*)parsec_dequeue_try_pop_front( (parsec_dequeue_t*)es->scheduler_object );
    *distance = 0;
    return context;
}

static int sched_gd_schedule(parsec_execution_stream_t* es,
                             parsec_task_t* new_context,
                             int32_t distance)
{
    if( (new_context->task_class->flags & PARSEC_HIGH_PRIORITY_TASK) &&
        (0 == distance) ) {
        parsec_dequeue_chain_front( (parsec_dequeue_t*)es->scheduler_object, (parsec_list_item_t*)new_context);
    } else {
        parsec_dequeue_chain_back( (parsec_dequeue_t*)es->scheduler_object, (parsec_list_item_t*)new_context);
    }
    return 0;
}

static void sched_gd_remove( parsec_context_t *master )
{
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    int p, t;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            OBJ_RELEASE( es->scheduler_object );
            es->scheduler_object = NULL;
        }
    }
}
