/**
 * Copyright (c) 2013-2014 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 * 
 * Additional copyrights may follow
 * 
 * $HEADER$
 *
 */

#include "dague_config.h"
#include "dague/dague_internal.h"
#include "dague/debug.h"
#include "dague/class/dequeue.h"

#include "dague/mca/sched/sched.h"
#include "dague/mca/sched/sched_local_queues_utils.h"
#include "dague/mca/sched/lfq/sched_lfq.h"
#include "dague/class/dequeue.h"
#include "dague/mca/pins/pins.h"
#include "dague/dague_hwloc.h"

static int SYSTEM_NEIGHBOR = 0;

#if defined(DAGUE_PROF_TRACE)
#define TAKE_TIME(EU_PROFILE, KEY, ID)  dague_profiling_trace((EU_PROFILE), (KEY), (ID), NULL)
#else
#define TAKE_TIME(EU_PROFILE, KEY, ID) do {} while(0)
#endif

/**
 * Module functions
 */
static int sched_lfq_install(dague_context_t* master);
static int sched_lfq_schedule(dague_execution_unit_t* eu_context, dague_execution_context_t* new_context);
static dague_execution_context_t *sched_lfq_select( dague_execution_unit_t *eu_context );
static void sched_lfq_remove(dague_context_t* master);
static int flow_lfq_init(dague_execution_unit_t* eu_context, struct dague_barrier_t* barrier);

const dague_sched_module_t dague_sched_lfq_module = {
    &dague_sched_lfq_component,
    {
        sched_lfq_install,
        flow_lfq_init,
        sched_lfq_schedule,
        sched_lfq_select,
        NULL,
        sched_lfq_remove
    }
};

static int sched_lfq_install( dague_context_t *master )
{
    SYSTEM_NEIGHBOR = master->nb_vp * master->virtual_processes[0]->nb_cores; // defined for instrumentation
    return 0;
}

static int flow_lfq_init(dague_execution_unit_t* eu_context, struct dague_barrier_t* barrier)
{
    local_queues_scheduler_object_t *sched_obj = NULL;
    int nq, hwloc_levels;
    uint32_t queue_size;
    dague_vp_t* vp;

    vp = eu_context->virtual_process;

    /* Every flow creates its own local object */
    sched_obj = (local_queues_scheduler_object_t*)malloc(sizeof(local_queues_scheduler_object_t));
    eu_context->scheduler_object = sched_obj;
    if( 0 == eu_context->th_id ) {  /* And flow 0 creates the system_queue */
        sched_obj->system_queue = OBJ_NEW(dague_dequeue_t);
    }

    sched_obj->nb_hierarch_queues = vp->nb_cores;
    sched_obj->hierarch_queues = (dague_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(dague_hbbuffer_t*) );
    queue_size = vp->nb_cores * 4;

    /* All local allocations are now completed. Synchronize with the other
     threads before setting up the entire queues hierarchy. */
    dague_barrier_wait(barrier);

    /* Get the flow 0 system queue and store it locally */
    sched_obj->system_queue = LOCAL_QUEUES_OBJECT(vp->execution_units[0])->system_queue;

    /* Each thread creates its own "local" queue, connected to the shared dequeue */
    sched_obj->task_queue = dague_hbbuffer_new( queue_size, 1, push_in_queue_wrapper,
                                                (void*)sched_obj->system_queue);
    sched_obj->hierarch_queues[0] = sched_obj->task_queue;

    /* All local allocations are now completed. Synchronize with the other
     threads before setting up the entire queues hierarchy. */
    dague_barrier_wait(barrier);

    nq = 1;
#if defined(HAVE_HWLOC)
    hwloc_levels = dague_hwloc_nb_levels();
#else
    hwloc_levels = -1;
#endif

    /* Handle the case when HWLOC is present but cannot compute the hierarchy,
     * as well as the case when HWLOC is missing.
     */
    if( hwloc_levels == -1 ) {
        for( ; nq < sched_obj->nb_hierarch_queues; nq++ ) {
            sched_obj->hierarch_queues[nq] =
                LOCAL_QUEUES_OBJECT(vp->execution_units[(eu_context->th_id + nq) % vp->nb_cores])->task_queue;
        }
#if defined(HAVE_HWLOC)
    } else {
        /* Then, they know about all other queues, from the closest to the farthest */
        for(int level = 0; level <= hwloc_levels; level++) {
            for(int id = (eu_context->th_id + 1) % vp->nb_cores;
                id != eu_context->th_id;
                id = (id + 1) %  vp->nb_cores) {
                int d;
                d = dague_hwloc_distance(eu_context->th_id, id);
                if( d == 2*level || d == 2*level + 1 ) {
                    sched_obj->hierarch_queues[nq] = LOCAL_QUEUES_OBJECT(vp->execution_units[id])->task_queue;
                    DEBUG(("%d of %d: my %d preferred queue is the task queue of %d (%p)\n",
                           eu_context->th_id, eu_context->virtual_process->vp_id, nq, id, sched_obj->hierarch_queues[nq]));
                    nq++;
                    if( nq == sched_obj->nb_hierarch_queues )
                        break;
                }
            }
            if( nq == sched_obj->nb_hierarch_queues )
                break;
        }
        assert( nq == sched_obj->nb_hierarch_queues );
#endif
    }
    return 0;
}

static dague_execution_context_t *sched_lfq_select( dague_execution_unit_t *eu_context )
{
    dague_execution_context_t *exec_context = NULL;
    int i;
    exec_context = (dague_execution_context_t*)dague_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->task_queue,
                                                                       dague_execution_context_priority_comparator);
    if( NULL != exec_context ) {
#if defined(PINS_ENABLE)
        exec_context->victim_core = LOCAL_QUEUES_OBJECT(eu_context)->task_queue->assoc_core_num;
#endif
        return exec_context;
    }
    for(i = 0; i <  LOCAL_QUEUES_OBJECT(eu_context)->nb_hierarch_queues; i++ ) {
        exec_context = (dague_execution_context_t*)dague_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i],
                                                                           dague_execution_context_priority_comparator);
        if( NULL != exec_context ) {
            DEBUG3(("LQ\t: %d:%d found task %p in its %d-preferred hierarchical queue %p\n",
                    eu_context->virtual_process->vp_id, eu_context->th_id, exec_context, i, LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i]));
#if defined(PINS_ENABLE)
            exec_context->victim_core = LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i]->assoc_core_num;
#endif
            return exec_context;
        }
    }

    exec_context = (dague_execution_context_t *)dague_dequeue_try_pop_front(LOCAL_QUEUES_OBJECT(eu_context)->system_queue);
    if( NULL != exec_context ) {
        DEBUG3(("LQ\t: %d:%d found task %p in its system queue %p\n",
                eu_context->virtual_process->vp_id, eu_context->th_id, exec_context, LOCAL_QUEUES_OBJECT(eu_context)->system_queue));
#if defined(PINS_ENABLE)
        exec_context->victim_core = SYSTEM_NEIGHBOR;
#endif
    }
    return exec_context;
}

static int sched_lfq_schedule( dague_execution_unit_t* eu_context,
                              dague_execution_context_t* new_context )
{
    dague_hbbuffer_push_all( LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)new_context );
/* #if defined(DAGUE_PROF_TRACE) */
/*     TAKE_TIME(eu_context->eu_profile, queue_add_begin, 0); */
/*     TAKE_TIME(eu_context->eu_profile, queue_add_end, 0); */
/* #endif */
    return 0;
}

static void sched_lfq_remove( dague_context_t *master )
{
    int p, t;
    dague_execution_unit_t *eu;
    dague_vp_t *vp;
    local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            if (eu != NULL) {
                sched_obj = LOCAL_QUEUES_OBJECT(eu);

                if( eu->th_id == 0 ) {
                    OBJ_DESTRUCT( sched_obj->system_queue );
                    free( sched_obj->system_queue );
                }
                sched_obj->system_queue = NULL;

                dague_hbbuffer_destruct( sched_obj->task_queue );
                sched_obj->task_queue = NULL;

                free(sched_obj->hierarch_queues);
                sched_obj->hierarch_queues = NULL;

                free(eu->scheduler_object);
                eu->scheduler_object = NULL;
            }
            // else the scheduler wasn't really initialized anyway
        }
    }
}
