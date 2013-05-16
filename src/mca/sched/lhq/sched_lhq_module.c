/*
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 *
 * These symbols are in a file by themselves to provide nice linker
 * semantics.  Since linkers generally pull in symbols by object
 * files, keeping these symbols as the only symbols in this file
 * prevents utility programs such as "ompi_info" from having to import
 * entire components just to query their version and parameters.
 */

#include "dague_config.h"
#include "dague_internal.h"
#include "debug.h"
#include "dequeue.h"

#include "dague/mca/sched/sched.h"
#include "dague/mca/sched/sched_local_queues_utils.h"
#include "dague/mca/sched/lhq/sched_lhq.h"
#include "dequeue.h"
#include "dague/mca/pins/pins.h"
static int SYSTEM_NEIGHBOR = 0;

#if defined(DAGUE_PROF_TRACE) && 0
#define TAKE_TIME(EU_PROFILE, KEY, ID)  dague_profiling_trace((EU_PROFILE), (KEY), (ID), NULL)
#else
#define TAKE_TIME(EU_PROFILE, KEY, ID) do {} while(0)
#endif

/*
 * Module functions
 */
static int sched_lhq_install(dague_context_t* master);
static int sched_lhq_schedule(dague_execution_unit_t* eu_context, dague_execution_context_t* new_context);
static dague_execution_context_t *sched_lhq_select( dague_execution_unit_t *eu_context );
static void sched_lhq_remove(dague_context_t* master);

const dague_sched_module_t dague_sched_lhq_module = {
    &dague_sched_lhq_component,
    {
        sched_lhq_install,
        sched_lhq_schedule,
        sched_lhq_select,
        NULL,
        sched_lhq_remove
    }
};

static int sched_lhq_install( dague_context_t *master )
{
    int p, t;
    dague_execution_unit_t *eu;
    dague_vp_t *vp;
    local_queues_scheduler_object_t *sched_obj = NULL;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];

            sched_obj = (local_queues_scheduler_object_t*)malloc(sizeof(local_queues_scheduler_object_t));
            eu->scheduler_object = sched_obj;

            if( eu->th_id == 0 ) {
                sched_obj->system_queue = (dague_dequeue_t*)malloc(sizeof(dague_dequeue_t));
                OBJ_CONSTRUCT(sched_obj->system_queue, dague_dequeue_t);
            } else {
                sched_obj->system_queue = LOCAL_QUEUES_OBJECT(vp->execution_units[0])->system_queue;
            }

            sched_obj->nb_hierarch_queues = vp->nb_cores;
            sched_obj->hierarch_queues = (dague_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(dague_hbbuffer_t*) );

            sched_obj->nb_hierarch_queues = dague_hwloc_nb_levels();
            sched_obj->hierarch_queues = (dague_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(dague_hbbuffer_t*) );

            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = dague_hwloc_master_id(level, eu->th_id);
                if( eu->th_id == m ) {
                    int nbcores = dague_hwloc_nb_cores(level, m);
                    int queue_size = 96 * (level+1) / nbcores;
                    if( queue_size < nbcores ) queue_size = nbcores;

                    /* The master(s) create the shared queues */
                    sched_obj->hierarch_queues[idx] =
                        dague_hbbuffer_new( queue_size, nbcores,
                                            level == 0 ? push_in_queue_wrapper : push_in_buffer_wrapper,
                                            level == 0 ? (void*)sched_obj->system_queue : (void*)sched_obj->hierarch_queues[idx+1]);
                    sched_obj->hierarch_queues[idx]->assoc_core_num = p * vp->nb_cores + t; // stored for PINS
                    DEBUG3(("schedHQ %d: \tcreates hbbuffer of size %d (ideal %d) for level %d stored in %d: %p (parent: %p -- %s)\n",
                            eu->th_id, queue_size, nbcores,
                            level, idx, sched_obj->hierarch_queues[idx],
                            level == 0 ? (void*)sched_obj->system_queue : (void*)sched_obj->hierarch_queues[idx+1],
                            level == 0 ? "System queue" : "upper level hhbuffer"));
                }

            }
        }

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = (local_queues_scheduler_object_t*)eu->scheduler_object;
            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = dague_hwloc_master_id(level, eu->th_id);
                if( eu->th_id != m ) {
                    DEBUG3(("schedHQ %d: \ttakes the buffer of %d at level %d stored in %d: %p\n",
                            eu->th_id, m, level, idx, LOCAL_QUEUES_OBJECT(vp->execution_units[m])->hierarch_queues[idx]));
                    /* The slaves take their queue for this level from their master */
                    sched_obj->hierarch_queues[idx] = LOCAL_QUEUES_OBJECT(vp->execution_units[m])->hierarch_queues[idx];
                }

            }
            sched_obj->task_queue = sched_obj->hierarch_queues[0];
        }
    }

    return 0;
}

static dague_execution_context_t *sched_lhq_select( dague_execution_unit_t *eu_context )
{
    dague_execution_context_t *exec_context = NULL;
    int i;
    PINS(SELECT_BEGIN, eu_context, NULL, NULL);

    exec_context = (dague_execution_context_t*)dague_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->task_queue,
                                                                       dague_execution_context_priority_comparator);
    if( NULL != exec_context ) {
		PINS(SELECT_END, eu_context, exec_context, (void *)LOCAL_QUEUES_OBJECT(eu_context)->task_queue->assoc_core_num);
        return exec_context;
    }
    for(i = 0; i <  LOCAL_QUEUES_OBJECT(eu_context)->nb_hierarch_queues; i++ ) {
        exec_context = (dague_execution_context_t*)dague_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i],
                                                                           dague_execution_context_priority_comparator);
        if( NULL != exec_context ) {
            DEBUG3(("LQ\t: %d:%d found task %p in its %d-preferred hierarchical queue %p\n",
                    eu_context->virtual_process->vp_id, eu_context->th_id, exec_context, i, LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i]));
			PINS(SELECT_END, eu_context, exec_context, (void *)LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i]->assoc_core_num);
            return exec_context;
        }
    }

    exec_context = (dague_execution_context_t *)dague_dequeue_try_pop_front(LOCAL_QUEUES_OBJECT(eu_context)->system_queue);
    if( NULL != exec_context ) {
        DEBUG3(("LQ\t: %d:%d found task %p in its system queue %p\n",
                eu_context->virtual_process->vp_id, eu_context->th_id, exec_context, LOCAL_QUEUES_OBJECT(eu_context)->system_queue));
    }
	PINS(SELECT_END, eu_context, exec_context, (void *)SYSTEM_NEIGHBOR);
    return exec_context;
}

static int sched_lhq_schedule( dague_execution_unit_t* eu_context,
                              dague_execution_context_t* new_context )
{
    dague_hbbuffer_push_all( LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)new_context );
#if defined(DAGUE_PROF_TRACE)
    TAKE_TIME(eu_context->eu_profile, queue_add_begin, 0);
    TAKE_TIME(eu_context->eu_profile, queue_add_end, 0);
#endif
    return 0;
}

static void sched_lhq_remove( dague_context_t *master )
{
    int p, t;
    dague_execution_unit_t *eu;
    dague_vp_t *vp;
    local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);

            for(int level = 0; level < sched_obj->nb_hierarch_queues; level++) {
                int idx = sched_obj->nb_hierarch_queues - 1 - level;
                int m = dague_hwloc_master_id(level, eu->th_id);
                if( eu->th_id == m ) {
                    dague_hbbuffer_destruct(sched_obj->hierarch_queues[idx]);
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
