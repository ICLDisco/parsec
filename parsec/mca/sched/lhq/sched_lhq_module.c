/**
 * Copyright (c) 2013-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "parsec_config.h"
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
#define TAKE_TIME(EU_PROFILE, KEY, ID)  PARSEC_PROFILING_TRACE((EU_PROFILE), (KEY), (ID), NULL)
#else
#define TAKE_TIME(EU_PROFILE, KEY, ID) do {} while(0)
#endif

/**
 * Module functions
 */
static int sched_lhq_install(parsec_context_t* master);
static int sched_lhq_schedule(parsec_execution_unit_t* eu_context, parsec_execution_context_t* new_context);
static parsec_execution_context_t *sched_lhq_select( parsec_execution_unit_t *eu_context );
static int flow_lhq_init(parsec_execution_unit_t* eu_context, struct parsec_barrier_t* barrier);
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

static int flow_lhq_init(parsec_execution_unit_t* eu_context, struct parsec_barrier_t* barrier)
{
    parsec_context_t *master = eu_context->virtual_process->parsec_context;
    local_queues_scheduler_object_t *sched_obj = NULL;
    parsec_execution_unit_t *eu;
    parsec_vp_t *vp;
    int p, t;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];

            sched_obj = (local_queues_scheduler_object_t*)malloc(sizeof(local_queues_scheduler_object_t));
            eu->scheduler_object = sched_obj;

            if( eu->th_id == 0 ) {
                sched_obj->system_queue = (parsec_dequeue_t*)malloc(sizeof(parsec_dequeue_t));
                OBJ_CONSTRUCT(sched_obj->system_queue, parsec_dequeue_t);
            } else {
                sched_obj->system_queue = LOCAL_QUEUES_OBJECT(vp->execution_units[0])->system_queue;
            }

            sched_obj->nb_hierarch_queues = vp->nb_cores;
            sched_obj->hierarch_queues = (parsec_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(parsec_hbbuffer_t*) );

            sched_obj->nb_hierarch_queues = parsec_hwloc_nb_levels();
            sched_obj->hierarch_queues = (parsec_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(parsec_hbbuffer_t*) );

            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = parsec_hwloc_master_id(level, eu->th_id);
                if( eu->th_id == m ) {
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
                            eu->th_id, queue_size, nbcores,
                            level, idx, sched_obj->hierarch_queues[idx],
                            level == 0 ? (void*)sched_obj->system_queue : (void*)sched_obj->hierarch_queues[idx+1],
                            level == 0 ? "System queue" : "upper level hhbuffer");
                }

            }
        }

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = (local_queues_scheduler_object_t*)eu->scheduler_object;
            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = parsec_hwloc_master_id(level, eu->th_id);
                if( eu->th_id != m ) {
                    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "schedHQ %d: \ttakes the buffer of %d at level %d stored in %d: %p",
                            eu->th_id, m, level, idx, LOCAL_QUEUES_OBJECT(vp->execution_units[m])->hierarch_queues[idx]);
                    /* The slaves take their queue for this level from their master */
                    sched_obj->hierarch_queues[idx] = LOCAL_QUEUES_OBJECT(vp->execution_units[m])->hierarch_queues[idx];
                }

            }
            sched_obj->task_queue = sched_obj->hierarch_queues[0];
        }
    }
    (void)barrier;
    return 0;
}

static parsec_execution_context_t *sched_lhq_select( parsec_execution_unit_t *eu_context )
{
    parsec_execution_context_t *exec_context = NULL;
    int i;

    exec_context = (parsec_execution_context_t*)parsec_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->task_queue,
                                                                       parsec_execution_context_priority_comparator);
    if( NULL != exec_context ) {
        return exec_context;
    }
    for(i = 0; i <  LOCAL_QUEUES_OBJECT(eu_context)->nb_hierarch_queues; i++ ) {
        exec_context = (parsec_execution_context_t*)parsec_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i],
                                                                           parsec_execution_context_priority_comparator);
        if( NULL != exec_context ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in its %d-preferred hierarchical queue %p",
                    eu_context->virtual_process->vp_id, eu_context->th_id, exec_context, i, LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i]);
            return exec_context;
        }
    }

    exec_context = (parsec_execution_context_t *)parsec_dequeue_try_pop_front(LOCAL_QUEUES_OBJECT(eu_context)->system_queue);
    if( NULL != exec_context ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in its system queue %p",
                eu_context->virtual_process->vp_id, eu_context->th_id, exec_context, LOCAL_QUEUES_OBJECT(eu_context)->system_queue);
    }
    return exec_context;
}

static int sched_lhq_schedule( parsec_execution_unit_t* eu_context,
                              parsec_execution_context_t* new_context )
{
    parsec_hbbuffer_push_all( LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (parsec_list_item_t*)new_context );
#if defined(PARSEC_PROF_TRACE)
    TAKE_TIME(eu_context->eu_profile, queue_add_begin, 0);
    TAKE_TIME(eu_context->eu_profile, queue_add_end, 0);
#endif
    return 0;
}

static void sched_lhq_remove( parsec_context_t *master )
{
    int p, t;
    parsec_execution_unit_t *eu;
    parsec_vp_t *vp;
    local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);

            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = parsec_hwloc_master_id(level, eu->th_id);
                if( eu->th_id == m ) {
                    parsec_hbbuffer_destruct(sched_obj->hierarch_queues[idx]);
                    sched_obj->hierarch_queues[idx] = NULL;
                } else {
                    sched_obj->hierarch_queues[idx] = NULL;
                }
            }

            sched_obj->task_queue = NULL;

            free(eu->scheduler_object);
            eu->scheduler_object = NULL;
        }
    }
}
