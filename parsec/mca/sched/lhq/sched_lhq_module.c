/**
 * Copyright (c) 2013-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2024      NVIDIA Corporation.  All rights reserved.
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
#include "parsec/class/dequeue.h"

#include "parsec/mca/sched/sched.h"
#include "parsec/mca/sched/sched_local_queues_utils.h"
#include "parsec/mca/sched/lhq/sched_lhq.h"
#include "parsec/class/dequeue.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/papi_sde.h"

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
    return PARSEC_SUCCESS;
}

static int flow_lhq_init(parsec_execution_stream_t* ces, struct parsec_barrier_t* barrier)
{
    parsec_mca_sched_local_queues_scheduler_object_t *sched_obj = NULL;
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    int t;

    vp = ces->virtual_process;

    /* Due to the complexity of the hierarchy, building the LHQ is centralized */
    if( ces->th_id == 0 ) {

        for(t = 0; t < vp->nb_cores; t++) {
            /* First of all, we allocate the scheduling object memory for all threads */
            es = vp->execution_streams[t];

            sched_obj = (parsec_mca_sched_local_queues_scheduler_object_t*)calloc(1, sizeof(parsec_mca_sched_local_queues_scheduler_object_t));
            es->scheduler_object = sched_obj;

            if( es->th_id == 0 ) {
                assert(t == 0);
                sched_obj->system_queue = (parsec_dequeue_t*)malloc(sizeof(parsec_dequeue_t));
                PARSEC_OBJ_CONSTRUCT(sched_obj->system_queue, parsec_dequeue_t);
            } else {
                assert(t > 0);
                sched_obj->system_queue = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[0])->system_queue;
            }

            sched_obj->nb_hierarch_queues = vp->nb_cores;
            sched_obj->hierarch_queues = (parsec_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(parsec_hbbuffer_t*) );

            sched_obj->nb_hierarch_queues = parsec_hwloc_nb_levels();
            sched_obj->hierarch_queues = (parsec_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(parsec_hbbuffer_t*) );
        }

        for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
            /* Now we work level of the hierarchy per level */
            int idx = sched_obj->nb_hierarch_queues - 1 - level;
            for(t = 0; t < vp->nb_cores; t++) {
                /* First, find which threads are a master for the lower level of the hierarchy,
                 * and create a hbbuffer for them */
                es = vp->execution_streams[t];
                sched_obj = (parsec_mca_sched_local_queues_scheduler_object_t*)es->scheduler_object;
                int m = parsec_hwloc_master_id(level, es->th_id);
                if( 0 > m ) parsec_fatal("lhq scheduler requires a working hwloc");
                if( es->th_id == m ) {
                    int nbcores = parsec_hwloc_nb_cores(level, m);
                    if( 0 > nbcores ) parsec_fatal("lhq scheduler requires a working hwloc");
                    int queue_size = 96 * (level+1) / nbcores;
                    if( queue_size < nbcores ) queue_size = nbcores;

                    /* The master(s) create the shared queues */
                    sched_obj->hierarch_queues[idx] =
                            parsec_hbbuffer_new( queue_size, nbcores,
                                                 level == 0 ? parsec_mca_sched_push_in_system_queue_wrapper : parsec_mca_sched_push_in_buffer_wrapper,
                                                 level == 0 ? (void*)sched_obj : (void*)sched_obj->hierarch_queues[idx+1]);
                    sched_obj->hierarch_queues[idx]->assoc_core_num = ces->virtual_process->vp_id * vp->nb_cores + t; // stored for PINS
                    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "schedHQ %d: \tcreates hbbuffer of size %d (ideal %d) for level %d stored in %d: %p (parent: %p -- %s)",
                                         es->th_id, queue_size, nbcores,
                                         level, idx, sched_obj->hierarch_queues[idx],
                                         level == 0 ? (void*)sched_obj->system_queue : (void*)sched_obj->hierarch_queues[idx+1],
                                         level == 0 ? "System queue" : "upper level hhbuffer");
                }
            }
            for(t = 0; t < vp->nb_cores; t++) {
                /* Now that the queues have been created for this level, the non-masters copy the queue */
                es = vp->execution_streams[t];
                sched_obj = (parsec_mca_sched_local_queues_scheduler_object_t*)es->scheduler_object;
                int m = parsec_hwloc_master_id(level, es->th_id);
                if(m != es->th_id) {
                    sched_obj->hierarch_queues[idx] = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[m])->hierarch_queues[idx];
                }
            }
        }
        for(t = 0; t < vp->nb_cores; t++) {
            /* Last, the default task queue is the first hierarch queue of each es */
            es = vp->execution_streams[t];
            sched_obj = (parsec_mca_sched_local_queues_scheduler_object_t*)es->scheduler_object;
            sched_obj->task_queue = sched_obj->hierarch_queues[0];
        }

#if defined(PARSEC_PAPI_SDE)
        {
            char event_name[PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN];
            int t;
            parsec_vp_t *vp;
            parsec_execution_stream_t *es;
            parsec_mca_sched_local_queues_scheduler_object_t* sched_obj;
            vp = ces->virtual_process;
        
            snprintf(event_name, 256, "SCHEDULER::PENDING_TASKS::QUEUE=%d/overflow::SCHED=LHQ",
                     vp->vp_id);
            parsec_papi_sde_register_fp_counter(event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                         PAPI_SDE_int, (papi_sde_fptr_t)parsec_mca_sched_system_queue_length, vp);
            parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
            parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS::SCHED=LHQ", PAPI_SDE_SUM);
            
            for(t = 0; t < vp->nb_cores; t++) {
                es = vp->execution_streams[t];
                sched_obj = (parsec_mca_sched_local_queues_scheduler_object_t*)es->scheduler_object;
                
                for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                    int idx = sched_obj->nb_hierarch_queues - 1 - level;
                    int m = parsec_hwloc_master_id(level, es->th_id);
                    assert(m >= 0);
                    if( es->th_id == m ) {
                        snprintf(event_name, 256, "SCHEDULER::PENDING_TASKS::QUEUE=%d/%d::SCHED=LHQ",
                                 es->virtual_process->vp_id, idx);
                        parsec_papi_sde_register_fp_counter(event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                                     PAPI_SDE_int, (papi_sde_fptr_t)parsec_hbbuffer_approx_occupency,
                                                     sched_obj->hierarch_queues[idx]);
                        parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
                        parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS::SCHED=LHQ", PAPI_SDE_SUM);
                    }
                }
            }
        }
#endif
    }

    /* All threads wait here until the main one has completed the build */
    parsec_barrier_wait(barrier);

    return PARSEC_SUCCESS;
}

static parsec_task_t*
sched_lhq_select(parsec_execution_stream_t *es,
                 int32_t* distance)
{
    parsec_task_t *task = NULL;
    int i;

    task = (parsec_task_t*)parsec_hbbuffer_pop_best(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->task_queue,
                                                    parsec_execution_context_priority_comparator);
    if( NULL != task ) {
        *distance = 0;
        return task;
    }
    for(i = 0; i <  PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->nb_hierarch_queues; i++ ) {
        task = (parsec_task_t*)parsec_hbbuffer_pop_best(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->hierarch_queues[i],
                                                        parsec_execution_context_priority_comparator);
        if( NULL != task ) {
            PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in its %d-preferred hierarchical queue %p",
                                 es->virtual_process->vp_id, es->th_id, task, i, PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->hierarch_queues[i]);
            *distance = i + 1;
            return task;
        }
    }

    task = parsec_mca_sched_pop_from_system_queue_wrapper(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es));
    if( NULL != task ) {
        PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "LQ\t: %d:%d found task %p in its system queue %p",
                             es->virtual_process->vp_id, es->th_id, task, PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->system_queue);
        *distance = 1 + PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->nb_hierarch_queues;
    }
    return task;
}

static int sched_lhq_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance)
{
    parsec_hbbuffer_push_all( PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->task_queue,
                              (parsec_list_item_t*)new_context,
                              distance );
    return PARSEC_SUCCESS;
}

static void sched_lhq_remove( parsec_context_t *master )
{
    int p, t;
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    parsec_mca_sched_local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            sched_obj = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es);

            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = parsec_hwloc_master_id(level, es->th_id);
                if( es->th_id == m ) {
                    parsec_hbbuffer_destruct(sched_obj->hierarch_queues[idx]);
                    sched_obj->hierarch_queues[idx] = NULL;
                    PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::QUEUE=%d/%d::SCHED=LHQ", vp->vp_id, idx);
                } else {
                    sched_obj->hierarch_queues[idx] = NULL;
                }
            }

            sched_obj->task_queue = NULL;

            free(es->scheduler_object);
            es->scheduler_object = NULL;
        }
        PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::QUEUE=%d/overflow::SCHED=LHQ", p);
    }
    PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::SCHED=LHQ");
}
