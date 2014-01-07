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
#include "dague/mca/sched/ltq/sched_ltq.h"
#include "dequeue.h"
#include "maxheap.h"
#include "dague/mca/pins/pins.h"

#define dague_heap_priority_comparator (offsetof(dague_heap_t, priority))
static int SYSTEM_NEIGHBOR = 0;

/*
 * Module functions
 */
static int sched_ltq_install(dague_context_t* master);
static int sched_ltq_schedule(dague_execution_unit_t* eu_context, dague_execution_context_t* new_context);
static dague_execution_context_t *sched_ltq_select( dague_execution_unit_t *eu_context );
static void sched_ltq_remove(dague_context_t* master);

const dague_sched_module_t dague_sched_ltq_module = {
    &dague_sched_ltq_component,
    {
        sched_ltq_install,
        NULL,
        sched_ltq_schedule,
        sched_ltq_select,
        NULL,
        sched_ltq_remove
    }
};

static int sched_ltq_install( dague_context_t *master )
{
    int t, p, nq = 1;
    dague_execution_unit_t *eu;
    dague_vp_t * vp;
    uint32_t queue_size;
    local_queues_scheduler_object_t *sched_obj = NULL;
    int hwloc_levels;

    SYSTEM_NEIGHBOR = master->nb_vp * master->virtual_processes[0]->nb_cores; // defined for instrumentation

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = (local_queues_scheduler_object_t*)malloc(sizeof(local_queues_scheduler_object_t));
            eu->scheduler_object = sched_obj;

            if( eu->th_id == 0 ) {
                sched_obj->system_queue = (dague_dequeue_t*)malloc(sizeof(dague_dequeue_t));
                OBJ_CONSTRUCT( sched_obj->system_queue, dague_dequeue_t );
            } else {
                sched_obj->system_queue = LOCAL_QUEUES_OBJECT(vp->execution_units[0])->system_queue;
            }
        }

        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);

            sched_obj->nb_hierarch_queues = vp->nb_cores;
            sched_obj->hierarch_queues = (dague_hbbuffer_t **)malloc(sched_obj->nb_hierarch_queues * sizeof(dague_hbbuffer_t*) );
            queue_size = vp->nb_cores * 4;

            /* Each thread creates its own "local" queue, connected to the shared dequeue */
            sched_obj->task_queue = dague_hbbuffer_new( queue_size, 1, push_in_queue_wrapper,
                                                        (void*)sched_obj->system_queue);
            sched_obj->task_queue->assoc_core_num = p * vp->nb_cores + t; // stored for PINS
            sched_obj->hierarch_queues[0] = sched_obj->task_queue;
        }

        for(t = 0; t < vp->nb_cores; t++) {
            nq = 1;
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);

#if defined(HAVE_HWLOC)
            hwloc_levels = dague_hwloc_nb_levels();
#else
            hwloc_levels = -1;
#endif

            /* Handle the case when HWLOC is present but cannot compute the hierarchy,
             * as well as the casewhen HWLOC is not present
             */
            if( hwloc_levels == -1 ) {
                for( ; nq < sched_obj->nb_hierarch_queues; nq++ ) {
                    sched_obj->hierarch_queues[nq] =
                        LOCAL_QUEUES_OBJECT(vp->execution_units[(eu->th_id + nq) % vp->nb_cores])->task_queue;
                }
            } else {

#if defined(HAVE_HWLOC)
                /* Then, they know about all other queues, from the closest to the farthest */
                for(int level = 0; level <= hwloc_levels; level++) {
                    for(int id = (eu->th_id + 1) % vp->nb_cores;
                        id != eu->th_id;
                        id = (id + 1) %  vp->nb_cores) {
                        int d;
                        d = dague_hwloc_distance(eu->th_id, id);
                        if( d == 2*level || d == 2*level + 1 ) {
                            sched_obj->hierarch_queues[nq] = LOCAL_QUEUES_OBJECT(vp->execution_units[id])->task_queue;
                            DEBUG(("%d of %d: my %d preferred queue is the task queue of %d (%p)\n",
                                   eu->th_id, eu->virtual_process->vp_id, nq, id, sched_obj->hierarch_queues[nq]));
                            nq++;
                            if( nq == sched_obj->nb_hierarch_queues )
                                break;
                        }
                    }
                    if( nq == sched_obj->nb_hierarch_queues )
                        break;
                }
                assert( nq == sched_obj->nb_hierarch_queues );
#else
                /* Unreachable code */
#endif
            }
        }
    }
    return 0;
}

static dague_execution_context_t *sched_ltq_select( dague_execution_unit_t *eu_context )
{
    dague_heap_t* heap = NULL;
    dague_heap_t* new_heap = NULL;
    dague_execution_context_t * exec_context = NULL;
    int i = 0;
    /*
     possible future improvement over using existing pop_best function:
     instead, i need to iterate manually over the buffer
     and choose a tree that has the highest value
     then take that task from that tree.
     */
    heap = (dague_heap_t*)dague_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->task_queue, dague_heap_priority_comparator);
    exec_context = heap_remove(&heap);
    if( NULL != heap ) {
        dague_hbbuffer_push_all(LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)heap);
    }
    if (exec_context != NULL) {
		exec_context->victim_core = LOCAL_QUEUES_OBJECT(eu_context)->task_queue->assoc_core_num;
        return exec_context;
    }

    // if we failed to find one in our queue
    for(i = 1; i <  LOCAL_QUEUES_OBJECT(eu_context)->nb_hierarch_queues; i++ ) {
        heap = (dague_heap_t*)dague_hbbuffer_pop_best(LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i], dague_heap_priority_comparator);
        exec_context = heap_split_and_steal(&heap, &new_heap);
        if( NULL != heap ) {
            if (NULL != new_heap) {
                /* turn two-element doubly-linked list
                 (the default side effect of heap_split_and_steal)
                 into two singleton doubly-linked lists
                 */
                heap->list_item.list_next = (dague_list_item_t*)heap;
                heap->list_item.list_prev = (dague_list_item_t*)heap;
                new_heap->list_item.list_next = (dague_list_item_t*)new_heap;
                new_heap->list_item.list_prev = (dague_list_item_t*)new_heap;

                // put new heap back in neighboring queue
                dague_hbbuffer_push_all(LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i], (dague_list_item_t*)new_heap);
            }
            // put old heap in our queue -- it's a singleton either way by this point
            dague_hbbuffer_push_all(LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)heap);
        }
        if (exec_context != NULL) {
			exec_context->victim_core = LOCAL_QUEUES_OBJECT(eu_context)->hierarch_queues[i]->assoc_core_num;
            return exec_context;
        }
    }

    // if nothing yet, then go to system queue
    heap = (dague_heap_t *)dague_dequeue_pop_front(LOCAL_QUEUES_OBJECT(eu_context)->system_queue);
    exec_context = heap_split_and_steal(&heap, &new_heap);
    if (heap != NULL)
        dague_hbbuffer_push_all(LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)heap);
	if (exec_context != NULL)
		exec_context->victim_core = SYSTEM_NEIGHBOR;
    return exec_context;
}

static int sched_ltq_schedule( dague_execution_unit_t* eu_context,
                               dague_execution_context_t* new_context )
{
    dague_execution_context_t * cur = new_context;
    dague_execution_context_t * next;
    dague_heap_t* heap = heap_create();
    dague_heap_t* first_h = heap;
    int matches = 0;
    int i, j;

    // do data_lookup
    cur->function->prepare_input(eu_context, cur);

    while (1) {
        // check next element before insertion, which destroys next and prev
        next = (dague_execution_context_t*)cur->list_item.list_next;
        assert(next != NULL);

        heap_insert(heap, cur);

        if (next == cur /* looped */ || next == new_context /* looped */) {
            break; // we're done
        }

        // compare data.... if we have at least one similar data item, then group
        next->function->prepare_input(eu_context, next);
        matches = 0;
        for (i = 0; i < MAX_PARAM_COUNT; i++) {
            for (j = 0; j < MAX_PARAM_COUNT; j++) {
                if (cur->data[i].data_in != NULL && cur->data[i].data_in == next->data[j].data_in)
                    matches++;
            }
        }

        cur = next;

        if (!matches) {
            // make new heap
            dague_heap_t * new_heap = heap_create();
            heap->list_item.list_next->list_prev = (dague_list_item_t*)new_heap;
            new_heap->list_item.list_prev = (dague_list_item_t*)heap;
            new_heap->list_item.list_next = (dague_list_item_t*)heap->list_item.list_next;
            heap->list_item.list_next = (dague_list_item_t*)new_heap;
            heap = new_heap;
        }
    }

    dague_hbbuffer_push_all( LOCAL_QUEUES_OBJECT(eu_context)->task_queue, (dague_list_item_t*)first_h );

    return 0;
}

static void sched_ltq_remove( dague_context_t *master )
{
    int t, p;
    dague_execution_unit_t *eu;
    dague_vp_t *vp;
    local_queues_scheduler_object_t *sched_obj;

    for(p = 0; p < master->nb_vp; p++) {
        vp = master->virtual_processes[p];
        for(t = 0; t < vp->nb_cores; t++) {
            eu = vp->execution_units[t];
            sched_obj = LOCAL_QUEUES_OBJECT(eu);

            if( eu->th_id == 0 ) {
                OBJ_DESTRUCT(sched_obj->system_queue);
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
    }
}


