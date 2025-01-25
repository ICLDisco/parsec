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
#include "parsec/mca/sched/ltq/sched_ltq.h"
#include "parsec/class/dequeue.h"
#include "parsec/maxheap.h"
#include "parsec/mca/pins/pins.h"
#include "parsec/parsec_hwloc.h"
#include "parsec/papi_sde.h"

#define parsec_heap_priority_comparator (offsetof(parsec_heap_t, priority))

/**
 * Module functions
 */
static int sched_ltq_install(parsec_context_t* master);
static int sched_ltq_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance);
static parsec_task_t *sched_ltq_select(parsec_execution_stream_t *es,
                                       int32_t* distance);
static int flow_ltq_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier);
static void sched_ltq_remove(parsec_context_t* master);

const parsec_sched_module_t parsec_sched_ltq_module = {
    &parsec_sched_ltq_component,
    {
        sched_ltq_install,
        flow_ltq_init,
        sched_ltq_schedule,
        sched_ltq_select,
        NULL,
        sched_ltq_remove
    }
};

static int sched_ltq_install( parsec_context_t *master )
{
    (void)master;
    return PARSEC_SUCCESS;
}

static int flow_ltq_init(parsec_execution_stream_t* es, struct parsec_barrier_t* barrier)
{
    parsec_mca_sched_local_queues_scheduler_object_t *sched_obj = NULL;
    int nq = 1, hwloc_levels;
    uint32_t queue_size;
    parsec_vp_t * vp = es->virtual_process;

    sched_obj = (parsec_mca_sched_local_queues_scheduler_object_t*)calloc(1, sizeof(parsec_mca_sched_local_queues_scheduler_object_t));
    es->scheduler_object = sched_obj;

    if( es->th_id == 0 ) {
        sched_obj->system_queue = (parsec_dequeue_t*)malloc(sizeof(parsec_dequeue_t));
        sched_obj->system_queue = PARSEC_OBJ_NEW(parsec_dequeue_t);
    }

    sched_obj->nb_hierarch_queues = vp->nb_cores;
    sched_obj->hierarch_queues = (parsec_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(parsec_hbbuffer_t*) );
    queue_size = vp->nb_cores * 4;

    /* All local allocations are now completed. Synchronize with the other
     threads before setting up the entire queues hierarchy. */
    parsec_barrier_wait(barrier);

    /* Get the flow 0 system queue and store it locally */
    sched_obj->system_queue = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[0])->system_queue;

    /* Each thread creates its own "local" queue, connected to the shared dequeue */
    sched_obj->task_queue = parsec_hbbuffer_new( queue_size, 1, parsec_mca_sched_push_in_system_queue_wrapper,
                                                (void*)sched_obj);
    sched_obj->task_queue->assoc_core_num = -1; // broken since flow added
    sched_obj->hierarch_queues[0] = sched_obj->task_queue;

    /* All local allocations are now completed. Synchronize with the other
     threads before setting up the entire queues hierarchy. */
    parsec_barrier_wait(barrier);

    nq = 1;
#if defined(PARSEC_HAVE_HWLOC)
    hwloc_levels = parsec_hwloc_nb_levels();
#else
    hwloc_levels = -1;
#endif

    /* Handle the case when HWLOC is present but cannot compute the hierarchy,
     * as well as the casewhen HWLOC is not present
     */
    if( hwloc_levels == -1 ) {
        for( ; nq < sched_obj->nb_hierarch_queues; nq++ ) {
            sched_obj->hierarch_queues[nq] =
                PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[(es->th_id + nq) % vp->nb_cores])->task_queue;
        }
#if defined(PARSEC_HAVE_HWLOC)
    }
    else {
        /* Then, they know about all other queues, from the closest to the farthest */
        for(int level = 0; level <= hwloc_levels; level++) {
            for(int id = (es->th_id + 1) % vp->nb_cores;
                id != es->th_id;
                id = (id + 1) %  vp->nb_cores) {
                int d;
                d = parsec_hwloc_distance(es->th_id, id);
                if( d == 2*level || d == 2*level + 1 ) {
                    sched_obj->hierarch_queues[nq] = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[id])->task_queue;
                    PARSEC_DEBUG_VERBOSE(20, parsec_debug_output, "%d of %d: my %d preferred queue is the task queue of %d (%p)",
                           es->th_id, es->virtual_process->vp_id, nq, id, sched_obj->hierarch_queues[nq]);
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

#if defined(PARSEC_PAPI_SDE)
    if( 0 == es->th_id ) {
        char event_name[PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN];
        int thid;
        parsec_vp_t *vp;

        snprintf(event_name, PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN, "SCHEDULER::PENDING_TASKS::QUEUE=%d/overflow::SCHED=LTQ", es->virtual_process->vp_id);
        parsec_papi_sde_register_fp_counter(event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                     PAPI_SDE_int, (papi_sde_fptr_t)parsec_mca_sched_system_queue_length, es->virtual_process);
        parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
        parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS::SCHED=LTQ", PAPI_SDE_SUM);
        vp = es->virtual_process;
        for(thid = 0; thid < vp->nb_cores; thid++) {
            snprintf(event_name, PARSEC_PAPI_SDE_MAX_COUNTER_NAME_LEN, "SCHEDULER::PENDING_TASKS::QUEUE=%d/%d::SCHED=LTQ", vp->vp_id, thid);
            parsec_papi_sde_register_fp_counter(event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT,
                                         PAPI_SDE_int, (papi_sde_fptr_t)parsec_hbbuffer_approx_occupency,
                                         PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(vp->execution_streams[thid])->task_queue);
            parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS", PAPI_SDE_SUM);
            parsec_papi_sde_add_counter_to_group(event_name, "SCHEDULER::PENDING_TASKS::SCHED=LTQ", PAPI_SDE_SUM);
        }
    }
#endif

    return PARSEC_SUCCESS;
}

static parsec_task_t*
sched_ltq_select(parsec_execution_stream_t *es,
                 int32_t* distance)
{
    parsec_heap_t* heap = NULL;
    parsec_heap_t* new_heap = NULL;
    parsec_task_t * task = NULL;
    int i = 0;
    /*
     possible future improvement over using existing pop_best function:
     instead, i need to iterate manually over the buffer
     and choose a tree that has the highest value
     then take that task from that tree.
     */
    heap = (parsec_heap_t*)parsec_hbbuffer_pop_best(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->task_queue,
                                                    parsec_heap_priority_comparator);
    task = heap_remove(&heap);
    if( NULL != heap ) {
        parsec_hbbuffer_push_all(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->task_queue,
                                 (parsec_list_item_t*)heap, 0);
    }
    if (task != NULL) {
        *distance = 1;
        return task;
    }

    // if we failed to find one in our queue
    for(i = 1; i <  PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->nb_hierarch_queues; i++ ) {
        heap = (parsec_heap_t*)parsec_hbbuffer_pop_best(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->hierarch_queues[i], parsec_heap_priority_comparator);
        task = heap_split_and_steal(&heap, &new_heap);
        if( NULL != heap ) {
            if (NULL != new_heap) {
                /* turn two-element doubly-linked list
                 (the default side effect of heap_split_and_steal)
                 into two singleton doubly-linked lists
                 */
                heap->list_item.list_next = (parsec_list_item_t*)heap;
                heap->list_item.list_prev = (parsec_list_item_t*)heap;
                new_heap->list_item.list_next = (parsec_list_item_t*)new_heap;
                new_heap->list_item.list_prev = (parsec_list_item_t*)new_heap;

                // put new heap back in neighboring queue
                parsec_hbbuffer_push_all(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->hierarch_queues[i],
                                         (parsec_list_item_t*)new_heap, 0);
            }
            // put old heap in our queue -- it's a singleton either way by this point
            parsec_hbbuffer_push_all(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->task_queue,
                                     (parsec_list_item_t*)heap, 0);
        }
        if (task != NULL) {
            *distance = i;
            return task;
        }
    }

    // if nothing yet, then go to system queue
    heap = (parsec_heap_t *)parsec_dequeue_pop_front(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->system_queue);
    task = heap_split_and_steal(&heap, &new_heap);
#if defined(PARSEC_PAPI_SDE)
    if( NULL != task ) {
        PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->local_system_queue_balance--;
    }
#endif
    if (heap != NULL) {
#if defined(PARSEC_PAPI_SDE)
        PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->local_system_queue_balance-= heap->size;
#endif
        parsec_hbbuffer_push_all(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->task_queue,
                                 (parsec_list_item_t*)heap, 0);
    }
    *distance = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->nb_hierarch_queues + 1;
    return task;
}

static int sched_ltq_schedule(parsec_execution_stream_t* es,
                              parsec_task_t* new_context,
                              int32_t distance)
{
    parsec_task_t * cur = new_context;
    parsec_task_t * next;
    parsec_heap_t* heap = heap_create();
    parsec_heap_t* first_h = heap;
    int matches = 0;
    int i, j;

    while (1) {
        // check next element before insertion, which destroys next and prev
        next = (parsec_task_t*)cur->super.list_next;
        assert(next != NULL);

        heap_insert(heap, cur);

        if (next == cur /* looped */ || next == new_context /* looped */) {
            break; // we're done
        }

        /**
         * Count how many common inputs are shared by 2 consecutive tasks. If we found
         * at least one identical input we group the 2 tasks in the same heap. Otherwise
         * the task create a new heap. In both cases the new task become the base task
         * for further comparaison.
         */
        for( matches = i = 0; i < cur->task_class->nb_flows; i++) {
            for (j = 0; j < next->task_class->nb_flows; j++) {
                if(cur->data[i].data_in == next->data[j].data_in)
                    matches++;
            }
        }

        cur = next;

        if (!matches) {
            // make new heap
            parsec_heap_t * new_heap = heap_create();
            heap->list_item.list_next->list_prev = (parsec_list_item_t*)new_heap;
            new_heap->list_item.list_prev = (parsec_list_item_t*)heap;
            new_heap->list_item.list_next = (parsec_list_item_t*)heap->list_item.list_next;
            heap->list_item.list_next = (parsec_list_item_t*)new_heap;
            heap = new_heap;
        }
    }

    /* Insert the prepared heap elements starting from the correct distance */
    parsec_hbbuffer_push_all(PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es)->task_queue,
                             (parsec_list_item_t*)first_h, distance);

    return PARSEC_SUCCESS;
}

static void sched_ltq_remove( parsec_context_t *master )
{
    int t, p;
    parsec_execution_stream_t *es;
    parsec_vp_t *vp;
    parsec_mca_sched_local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            es = vp->execution_streams[t];
            sched_obj = PARSEC_MCA_SCHED_LOCAL_QUEUES_OBJECT(es);

            if( es->th_id == 0 ) {
                PARSEC_OBJ_DESTRUCT(sched_obj->system_queue);
                free( sched_obj->system_queue );
            }
            sched_obj->system_queue = NULL;

            parsec_hbbuffer_destruct( sched_obj->task_queue );
            sched_obj->task_queue = NULL;

            free(sched_obj->hierarch_queues);
            sched_obj->hierarch_queues = NULL;

            free(es->scheduler_object);
            es->scheduler_object = NULL;

            PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::QUEUE=%d/%d::SCHED=LTQ", vp->vp_id, t);
        }
        PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::QUEUE=%d/overflow::SCHED=LTQ", p);
    }
    PARSEC_PAPI_SDE_UNREGISTER_COUNTER("SCHEDULER::PENDING_TASKS::SCHED=LTQ");
}
