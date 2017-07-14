/**
 * Copyright (c) 2013-2017 The University of Tennessee and The University
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
#include "parsec/debug.h"
#include "parsec/class/dequeue.h"

#include "parsec/mca/sched/sched.h"
#include "parsec/mca/sched/sched_local_queues_utils.h"
#include "parsec/mca/sched/lhq/sched_lhq.h"
#include "parsec/class/dequeue.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/parsec_hwloc.h"

#if defined(PARSEC_PROF_TRACE) && 0
#define TAKE_TIME(ES_PROFILE, KEY, ID)  PARSEC_PROFILING_TRACE((ES_PROFILE), (KEY), (ID), NULL)
#else
#define TAKE_TIME(ES_PROFILE, KEY, ID) do {} while(0)
#endif

/**
 * Module functions
 */
static int sched_lhq_install(parsec_context_t* master);
static int sched_lhq_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance);
static parsec_task_t*
sched_lhq_select(parsec_execution_stream_t *es,
                 int32_t* distance);
static int flow_lhq_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier);
static void sched_lhq_remove(parsec_context_t* master);

const parsec_sched_module_t parsec_sched_lhq_module = {
    &parsec_sched_lhq_component,
    {
        sched_lhq_install,
        flow_lhq_init,
        sched_lhq_schedule,
        sched_lhq_select,
        NULL,
        sched_lhq_remove
    }
};

static int sched_lhq_install( parsec_context_t *master )
{
    (void)master;
    return 0;
}

static int flow_lhq_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier)
{
    parsec_context_t *master = es->virtual_process->parsec_context;
    local_queues_scheduler_object_t *sched_obj = NULL;
    parsec_vp_t *vp;
    int p, t;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];

            sched_obj = (local_queues_scheduler_object_t*)malloc(sizeof(local_queues_scheduler_object_t));
            es->scheduler_object = sched_obj;

            if( es->th_id == 0 ) {
                sched_obj->system_queue = (parsec_dequeue_t*)malloc(sizeof(parsec_dequeue_t));
                OBJ_CONSTRUCT(sched_obj->system_queue, parsec_dequeue_t);
            } else {
                sched_obj->system_queue = LOCAL_QUEUES_OBJECT(vp->execution_streams[0])->system_queue;
            }

            sched_obj->nb_hierarch_queues = vp->nb_cores;
            sched_obj->hierarch_queues = (parsec_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(parsec_hbbuffer_t*) );

            sched_obj->nb_hierarch_queues = parsec_hwloc_nb_levels();
            sched_obj->hierarch_queues = (parsec_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(parsec_hbbuffer_t*) );

            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = parsec_hwloc_master_id(level, es->th_id);
                assert(m >= 0);
                if( es->th_id == m ) {
                    int nbcores = parsec_hwloc_nb_cores(level, m);
                    int queue_size = 96 * (level+1) / nbcores;
                    if( queue_size < nbcores ) queue_size = nbcores;

                    /* The master(s) create the shared queues */
                    sched_obj->hierarch_queues[idx] =
                        parsec_hbbuffer_new( queue_size, nbcores,
                                            level == 0 ? push_in_queue_wrapper : push_in_buffer_wrapper,
                                            level == 0 ? (void*)sched_obj->system_queue : (void*)sched_obj->hierarch_queues[idx+1]);
                    sched_obj->hierarch_queues[idx]->assoc_core_num = p * vp->nb_cores + t; // stored for PINS
                    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "schedHQ %d: \tcreates hbbuffer of size %d (ideal %d) for level %d stored in %d: %p (parent: %p -- %s)",
                            es->th_id, queue_size, nbcores,
                            level, idx, sched_obj->hierarch_queues[idx],
                            level == 0 ? (void*)sched_obj->system_queue : (void*)sched_obj->hierarch_queues[idx+1],
                            level == 0 ? "System queue" : "upper level hhbuffer");
                }

            }
        }

        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            sched_obj = (local_queues_scheduler_object_t*)es->scheduler_object;
            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = parsec_hwloc_master_id(level, es->th_id);
                assert(m >= 0);
                if( es->th_id != m ) {
                    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "schedHQ %d: \ttakes the buffer of %d at level %d stored in %d: %p",
                            es->th_id, m, level, idx, LOCAL_QUEUES_OBJECT(vp->execution_streams[m])->hierarch_queues[idx]);
                    /* The slaves take their queue for this level from their master */
                    sched_obj->hierarch_queues[idx] = LOCAL_QUEUES_OBJECT(vp->execution_streams[m])->hierarch_queues[idx];
                }

            }
            sched_obj->task_queue = sched_obj->hierarch_queues[0];
        }
    }
    (void)barrier;
    return 0;
}

static parsec_task_t*
sched_lhq_select(parsec_execution_stream_t *es,
                 int32_t* distance)
{
    parsec_task_t *task = NULL;
    int i;

    task = (parsec_task_t*)parsec_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(es)->task_queue,
                                                                       parsec_execution_context_priority_comparator);
    if( NULL != task ) {
        *distance = 0;
        return task;
    }
    for(i = 0; i <  LOCAL_QUEUES_OBJECT(es)->nb_hierarch_queues; i++ ) {
        task = (parsec_task_t*)parsec_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(es)->hierarch_queues[i],
                                                                           parsec_execution_context_priority_comparator);
        if( NULL != task ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in its %d-preferred hierarchical queue %p",
                    es->virtual_process->vp_id, es->th_id, task, i, LOCAL_QUEUES_OBJECT(es)->hierarch_queues[i]);
            *distance = i + 1;
            return task;
        }
    }

    task = (parsec_task_t *)parsec_dequeue_try_pop_front(LOCAL_QUEUES_OBJECT(es)->system_queue);
    if( NULL != task ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in its system queue %p",
                es->virtual_process->vp_id, es->th_id, task, LOCAL_QUEUES_OBJECT(es)->system_queue);
        *distance = 1 + LOCAL_QUEUES_OBJECT(es)->nb_hierarch_queues;
    }
    return task;
}

static int sched_lhq_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance)
{
    parsec_hbbuffer_push_all( LOCAL_QUEUES_OBJECT(es)->task_queue,
                              (parsec_list_item_t*)new_context,
                              distance );
#if defined(PARSEC_PROF_TRACE)
    TAKE_TIME(es->es_profile, queue_add_begin, 0);
    TAKE_TIME(es->es_profile, queue_add_end, 0);
#endif
    return 0;
}

static void sched_lhq_remove( parsec_context_t *master )
{
    int p, t;
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            sched_obj = LOCAL_QUEUES_OBJECT(es);

            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = parsec_hwloc_master_id(level, es->th_id);
                if( es->th_id == m ) {
                    parsec_hbbuffer_destruct(sched_obj->hierarch_queues[idx]);
                    sched_obj->hierarch_queues[idx] = NULL;
                } else {
                    sched_obj->hierarch_queues[idx] = NULL;
                }
            }

            sched_obj->task_queue = NULL;

            free(es->scheduler_object);
            es->scheduler_object = NULL;
        }
    }
}
