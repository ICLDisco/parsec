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
#include "parsec/mca/sched/rnd/sched_rnd.h"
#include "parsec/class/dequeue.h"
#include "parsec/mca/pins/pins.h"

/**
 * Module functions
 */
static int sched_rnd_install(parsec_context_t* master);
static int sched_rnd_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance);
static parsec_task_t *sched_rnd_select(parsec_execution_stream_t *es,
                                       int32_t* distance);
static int flow_rnd_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier);
static void sched_rnd_remove(parsec_context_t* master);

const parsec_sched_module_t parsec_sched_rnd_module = {
    &parsec_sched_rnd_component,
    {
        sched_rnd_install,
        flow_rnd_init,
        sched_rnd_schedule,
        sched_rnd_select,
        NULL,
        sched_rnd_remove
    }
};

static int sched_rnd_install( parsec_context_t *master )
{
    (void)master;
    return 0;
}

static int flow_rnd_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier)
{
    parsec_vp_t *vp = es->virtual_process;

    if (es == vp->execution_streams[0])
        vp->execution_streams[0]->scheduler_object = OBJ_NEW(parsec_list_t);

    parsec_barrier_wait(barrier);

    es->scheduler_object = (void*)vp->execution_streams[0]->scheduler_object;

    return 0;
}

static parsec_task_t*
sched_rnd_select(parsec_execution_stream_t *es,
                 int32_t* distance)
{
    parsec_task_t * context =
        (parsec_task_t*)parsec_list_pop_front((parsec_list_t*)es->scheduler_object);
    *distance = 0;
    return context;
}

static int sched_rnd_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance)
{
    parsec_list_item_t *it = (parsec_list_item_t*)new_context;
#if defined(PARSEC_DEBUG_NOISIER)
    char tmp[MAX_TASK_STRLEN];
#endif
    do {
#if defined(PARSEC_DEBUG_NOISIER)
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "RND:\t Pushing task %s",
                parsec_task_snprintf(tmp, MAX_TASK_STRLEN, (parsec_task_t*)it));
#endif
        /* randomly assign priority */
        (*((int*)(((uintptr_t)it)+parsec_execution_context_priority_comparator))) = rand();
        it = (parsec_list_item_t*)((parsec_list_item_t*)it)->list_next;
    } while( it != (parsec_list_item_t*)new_context );
    parsec_list_chain_sorted((parsec_list_t*)es->scheduler_object,
                            (parsec_list_item_t*)new_context,
                            parsec_execution_context_priority_comparator);
    /* We can ignore distance, the task will randomly get inserted in a place that
     * will prevent livelocks. */
    (void)distance;
    return 0;
}

static void sched_rnd_remove( parsec_context_t *master )
{
    int p, t;
    parsec_vp_t *vp;
    parsec_execution_stream_t *es;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            if( es->th_id == 0 ) {
                OBJ_DESTRUCT( es->scheduler_object );
                free(es->scheduler_object);
            }
            es->scheduler_object = NULL;
        }
    }
}
